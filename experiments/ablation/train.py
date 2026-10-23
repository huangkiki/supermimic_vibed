"""Native simulator experiment runner. See docs/ablation/protocol.md.

Run from the repository root: python -m experiments.ablation.train --config ... --output ...
The original supermimic/train.py and its defaults are deliberately untouched.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

STARTED = time.perf_counter()

import torch
from torch import nn

from .common import METHODS, Config, Ledger, append_csv, freeze, identity, motion_files, write_json
from .ppo import GaussianActor, gae, ppo_update


class Experiment:
    def __init__(self, cfg, env, mdp, motion, ledger, manifest):
        from supermimic.buffer import ReplayBuffer
        from supermimic.mdp import CRITIC_EXTRA, INV_DIM, OBS_DIM
        from supermimic.networks import Critic, Policy, WorldModel

        self.cfg, self.env, self.mdp, self.motion, self.ledger = cfg, env, mdp, motion, ledger
        self.device = "cuda:0"
        torch.manual_seed(cfg.seed + 111)
        self.wm = WorldModel().to(self.device) if cfg.method != "ppo_sim" else None
        torch.manual_seed(cfg.seed + 222)
        self.pi = Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist).to(self.device)
        self.actor = GaussianActor(self.pi, initial_std=cfg.initial_std).to(self.device)
        torch.manual_seed(cfg.seed + 333)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * cfg.goal_frames).to(self.device)
        self.ppo = cfg.method.startswith("ppo")
        self.frozen = "_frozen" in cfg.method
        self.opt_pi = torch.optim.Adam(self.actor.parameters() if self.ppo else self.pi.parameters(),
                                       lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr) if self.wm else None
        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, self.device) if self.wm else None
        self.history = torch.zeros(cfg.num_envs, cfg.hist, OBS_DIM, device=self.device)
        self.wm_rollout = torch.compile(self._wm_rollout) if cfg.compile else self._wm_rollout
        self.imag_step = torch.compile(self._imag_step) if cfg.compile else self._imag_step
        self.reward_term = torch.compile(self._reward_term) if cfg.compile else self._reward_term
        self.critic_input = torch.compile(self._critic_input) if cfg.compile else self._critic_input
        if cfg.bootstrap:
            ck = torch.load(cfg.bootstrap, map_location=self.device, weights_only=False)
            if ck["config"]["method"] != "pretrain":
                raise ValueError("bootstrap must come from a model-only pretrain run")
            for key in ("motions", "assets"):
                if ck["manifest"][key] != manifest[key]:
                    raise ValueError(f"bootstrap identity mismatch: {key}")
            receipt = json.loads((Path(cfg.bootstrap).parent / "completion.json").read_text())
            if receipt["status"] != "completed" or receipt["iterations"] < 1:
                raise ValueError("bootstrap pretraining did not complete its declared budget")
            for key in ("seed", "subset", "num_envs", "buffer_steps", "hist", "goal_frames", "amp"):
                if ck["config"][key] != getattr(cfg, key):
                    raise ValueError(f"bootstrap mismatch: {key}")
            self.wm.load_state_dict(ck["wm"])
            self.pi.load_state_dict(ck["pi"])
            self.critic.load_state_dict(ck["critic"])
            self.opt_wm.load_state_dict(ck["opt_wm"])
            for key, value in ck["replay"]["data"].items():
                self.buffer.data[key].copy_(value)
            self.buffer.ptr, self.buffer.size = ck["replay"]["ptr"], ck["replay"]["size"]
            self.ledger.inherited = receipt["ledger"]
        torch.manual_seed(cfg.seed + 1000)

    def _critic_input(self, state, frames, xi_n, goal):
        return torch.cat((self.mdp.critic_features(state, frames, xi_n), goal), -1)

    def _reward_term(self, state, frames, action, previous):
        return self.mdp.reward(state, frames, action, previous), self.mdp.soft_termination(state, frames)

    def _imag_step(self, state, history, frames, xi, xi_n, goal):
        action = self.pi(history, goal)
        next_state = self.wm.step(state, action, xi_n)
        frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, xi, frames))
        return next_state, action, torch.cat((history[:, 1:], obs[:, None]), 1), frames, self.motion.goal(frames, self.cfg.goal_frames)

    def _wm_rollout(self, states, actions, xi):
        from supermimic.mdp import normalize_xi, unflatten_state
        state = unflatten_state(states[:, 0])
        predictions = []
        for j in range(self.cfg.wm_window):
            state = self.wm.step(state, actions[:, j], normalize_xi(xi[:, j]))
            predictions.append(state)
        return tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))

    @torch.no_grad()
    def observation(self):
        obs = self.mdp.add_noise(self.mdp.obs(self.env.state(), self.env.a_prev, self.env.xi, self.env.f))
        self.history = torch.where((self.env.ep_step == 0)[:, None, None],
                                   obs[:, None].expand_as(self.history),
                                   torch.cat((self.history[:, 1:], obs[:, None]), 1))
        return obs, self.motion.goal(self.env.f, self.cfg.goal_frames)

    @torch.no_grad()
    def collect_model_data(self, steps):
        from supermimic.mdp import normalize_xi, unflatten_state
        with self.ledger.phase("collection"):
            for _ in range(steps):
                obs, goal = self.observation()
                self.pi.obs_norm.update(obs)
                self.pi.goal_norm.update(goal)
                if self.ppo:
                    action, _ = self.actor.sample(self.history, goal)
                else:
                    action = self.pi(self.history, goal)
                    action += self.cfg.initial_std * torch.randn_like(action)
                # Identical unclipped exploration convention for all experimental methods.
                out = self.env.step(action)
                self.ledger.simulator_steps += self.cfg.num_envs
                if self.buffer:
                    self.buffer.add(out)
                    s, s1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                    self.wm.in_norm.update(self.wm.inputs(s, action, normalize_xi(out["xi"])))
                    self.wm.out_norm.update(self.wm.targets(s, s1))

    def fit_world_model(self):
        from supermimic.mdp import unflatten_state
        floor = torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3
                             + [3e-3] * 29 + [5e-2] * 29, device=self.device)
        losses = []
        with self.ledger.phase("wm_training"):
            for _ in range(self.cfg.wm_batches):
                batch = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
                prediction = self.wm_rollout(batch["s"], batch["a"], batch["xi"])
                err = self.wm.state_error(prediction, unflatten_state(batch["s1"]))
                error = (err / torch.maximum(self.wm.out_norm.std(), floor)).square().mean(-1)
                loss = (error * batch["mask"]).sum() / self.cfg.wm_batch
                self.opt_wm.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0, error_if_nonfinite=True)
                self.opt_wm.step()
                losses.append(float(loss.detach()) / self.cfg.wm_window)
            self.opt_wm.zero_grad(set_to_none=True)
        return sum(losses) / len(losses)

    def sample_starts(self):
        wanted = self.cfg.pi_batch
        starts, histories, count = [], [], 0
        # Bounded rejection avoids an infinite loop on too-short/mis-indexed motion libraries.
        for _ in range(100):
            s, h = self.buffer.sample_starts(2 * wanted, self.cfg.hist)
            valid = s["f"] + self.cfg.horizon <= self.motion.clip_end[self.motion.clip_id[s["f"]]]
            ids = valid.nonzero().flatten()[:wanted - count]
            starts.append({k: v[ids] for k, v in s.items()})
            histories.append({k: v[ids] for k, v in h.items()})
            count += len(ids)
            if count == wanted:
                return ({k: torch.cat([d[k] for d in starts]) for k in s},
                        {k: torch.cat([d[k] for d in histories]) for k in h})
        raise RuntimeError("No sufficiently long replay windows for imagination")

    def imagine(self):
        from supermimic.mdp import normalize_xi, unflatten_state
        cfg = self.cfg
        freeze(self.wm, True)
        freeze(self.critic, True)
        with self.ledger.phase("imagined_rollout"), torch.set_grad_enabled(not self.ppo):
            start, hist = self.sample_starts()
            state = unflatten_state(start["s"])
            frames, previous, xi = start["f"], start["a_prev"], start["xi"]
            xi_n = normalize_xi(xi)
            history = self.mdp.add_noise(self.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]))
            goal = self.motion.goal(frames, cfg.goal_frames)
            initial_x = self.critic_input(state, frames, xi_n, goal)
            # Same update schedule for both optimizers; frozen during each rollout and all epochs.
            self.critic.norm.update(initial_x.detach())
            observations, goals, actions, old_logp, features, rewards, survival = [], [], [], [], [], [], []
            for _ in range(cfg.horizon):
                observations.append(history.detach())
                goals.append(goal.detach())
                features.append(self.critic_input(state, frames, xi_n, goal))
                if self.ppo:
                    action, lp = self.actor.sample(history, goal)
                    state = self.wm.step(state, action, xi_n)
                    frames = frames + 1
                    obs = self.mdp.add_noise(self.mdp.obs(state, action, xi, frames))
                    history = torch.cat((history[:, 1:], obs[:, None]), 1)
                    goal = self.motion.goal(frames, cfg.goal_frames)
                    old_logp.append(lp)
                else:
                    state, action, history, frames, goal = self.imag_step(state, history, frames, xi, xi_n, goal)
                reward, done_prob = self.reward_term(state, frames, action, previous)
                if self.ppo and cfg.ppo_wm_survival == "hard":
                    done_prob = self.mdp.termination(state, frames).float()
                actions.append(action)
                rewards.append(reward)
                survival.append(1 - done_prob)
                previous = action
            features.append(self.critic_input(state, frames, xi_n, goal))
            features, rewards, survival = torch.stack(features), torch.stack(rewards), torch.stack(survival)
            values = self.critic(features)
            alive = torch.cat((torch.ones_like(survival[:1]), survival.cumprod(0)), 0)
            discounts = cfg.gamma ** torch.arange(cfg.horizon, device=self.device)[:, None]
            weights = discounts * alive[:-1]
            objective = (weights * rewards).sum(0) + cfg.gamma ** cfg.horizon * alive[-1] * values[-1]
            advantages, targets = gae(rewards, values[:-1], values[1:], survival, survival, cfg.gamma, cfg.lam)
            batch = dict(obs=torch.stack(observations), goal=torch.stack(goals), action=torch.stack(actions),
                         critic_input=features[:-1], target=targets, advantage=advantages, weight=weights)
            if self.ppo:
                batch["log_prob"] = torch.stack(old_logp)
            self.ledger.imagined_steps += cfg.horizon * cfg.pi_batch
        return objective, batch

    def optimize_imagined(self):
        objective, batch = self.imagine()
        cfg = self.cfg
        with self.ledger.phase("policy_optimization"):
            if not self.ppo:
                loss = -objective.mean()
                self.opt_pi.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0, error_if_nonfinite=True)
                nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
                self.opt_pi.step()
            freeze(self.critic, False)
            if self.ppo:
                result = ppo_update(self.actor, self.critic, self.opt_pi, self.opt_v, batch,
                                    epochs=cfg.ppo_epochs, minibatches=cfg.minibatches, clip=cfg.clip,
                                    entropy_coef=cfg.entropy_coef, target_kl=cfg.target_kl)
                if any(p.grad is not None for p in self.wm.parameters()):
                    raise RuntimeError("PPO policy loss reached world-model weights")
            else:
                x = batch["critic_input"].detach().flatten(0, 1)
                y = batch["target"].detach().flatten()
                for _ in range(cfg.ppo_epochs):
                    for ids in torch.randperm(len(y), device=self.device).tensor_split(cfg.minibatches):
                        if not len(ids):
                            continue
                        value_loss = (self.critic(x[ids]) - y[ids]).square().mean()
                        self.opt_v.zero_grad(set_to_none=True)
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
                        self.opt_v.step()
                result = dict(policy_loss=float(loss.detach()), value_loss=float(value_loss.detach()),
                              approx_kl=0., actor_updates=1, critic_updates=cfg.ppo_epochs * cfg.minibatches, kl_stopped=False)
            freeze(self.wm, False)
        return result

    @torch.no_grad()
    def simulator_rollout(self):
        from supermimic.mdp import normalize_xi, unflatten_state
        cfg = self.cfg
        rows = {k: [] for k in ("obs", "goal", "action", "log_prob", "critic_input", "value", "next_value", "reward", "bootstrap", "trace")}
        with self.ledger.phase("collection"):
            for _ in range(cfg.horizon):
                _, goal = self.observation()
                x = self.critic_input(self.env.state(), self.env.f, normalize_xi(self.env.xi), goal)
                action, logp = self.actor.sample(self.history, goal)
                out = self.env.step(action)
                # env.step resets internally; use returned PRE-reset s1, f, xi for bootstrap.
                next_goal = self.motion.goal(out["f"] + 1, cfg.goal_frames)
                next_x = self.critic_input(unflatten_state(out["s1"]), out["f"] + 1,
                                           normalize_xi(out["xi"]), next_goal)
                row = dict(obs=self.history.clone(), goal=goal, action=action, log_prob=logp,
                           critic_input=x, value=self.critic(x), next_value=self.critic(next_x), reward=out["r"],
                           bootstrap=(~out["term"]).float(), trace=(~out["done"]).float())
                for key, value in row.items():
                    rows[key].append(value)
                self.ledger.simulator_steps += cfg.num_envs
        batch = {k: torch.stack(v) for k, v in rows.items()}
        batch["advantage"], batch["target"] = gae(batch["reward"], batch["value"], batch["next_value"],
                                                   batch["bootstrap"], batch["trace"], cfg.gamma, cfg.lam)
        batch["weight"] = torch.ones_like(batch["reward"])
        return batch

    def optimize_simulator(self):
        cfg = self.cfg
        batch = self.simulator_rollout()
        with self.ledger.phase("policy_optimization"):
            result = ppo_update(self.actor, self.critic, self.opt_pi, self.opt_v, batch,
                                epochs=cfg.ppo_epochs, minibatches=cfg.minibatches, clip=cfg.clip,
                                entropy_coef=cfg.entropy_coef, target_kl=cfg.target_kl)
            # Only change normalizers AFTER the last reuse of old log probabilities/values.
            self.pi.obs_norm.update(batch["obs"][:, :, -1])
            self.pi.goal_norm.update(batch["goal"])
            self.critic.norm.update(batch["critic_input"])
        return result

    def save(self, path, iteration, manifest, include_replay=False):
        data = dict(config=asdict(self.cfg), iteration=iteration, pi=self.pi.state_dict(),
                    log_std=self.actor.log_std.detach(), critic=self.critic.state_dict(),
                    wm=self.wm.state_dict() if self.wm else None, ledger=self.ledger.snapshot(),
                    manifest=manifest, opt_pi=self.opt_pi.state_dict(), opt_v=self.opt_v.state_dict(),
                    opt_wm=self.opt_wm.state_dict() if self.opt_wm else None)
        if include_replay:
            data["replay"] = dict(data=self.buffer.data, ptr=self.buffer.ptr, size=self.buffer.size)
        temp = path.with_suffix(".tmp")
        torch.save(data, temp)
        temp.replace(path)


def main():
    # Imports/setup, asset and data loading, compilation, calibration and warm-up are charged.
    started = STARTED
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap")
    add_launcher_args(parser)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    for key in ("method", "seed", "bootstrap"):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    cfg = Config(**raw).validate()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    import supermimic.networks as networks
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot import default_joint_pos_list
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
    torch.manual_seed(cfg.seed)
    manifest = identity(cfg)
    write_json(output / "manifest.json", manifest)
    ledger = Ledger(started)
    kin = G1Kinematics("cuda:0")
    motion = MotionLib([str(p) for p in motion_files(cfg)], kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    frozen = "_frozen" in cfg.method
    sim_cfg = None if frozen else make_sim("cuda:0", args)
    context = nullcontext() if frozen else launch_simulation(sim_cfg, args)
    with context:
        env = None if frozen else TrackingSim(sim_cfg, cfg.num_envs, motion, mdp, gamma=cfg.gamma)
        experiment = Experiment(cfg, env, mdp, motion, ledger, manifest)
        if not cfg.bootstrap:
            experiment.collect_model_data(cfg.warmup_steps)
        experiment.save(output / "ckpt_000000.pt", 0, manifest)
        status = "completed"
        completed = 0
        step_accumulator = 0.
        for i in range(1, cfg.iterations + 1):
            if time.perf_counter() - started >= cfg.max_wall_seconds:
                status = "budget_exhausted"
                break
            fraction = min((i - 1) / cfg.lr_decay_iters, 1.)
            lr = cfg.lr * (.03 + .97 * .5 * (1 + math.cos(math.pi * fraction)))
            for optimizer in (experiment.opt_pi, experiment.opt_v, experiment.opt_wm):
                if optimizer:
                    for group in optimizer.param_groups:
                        group["lr"] = lr
            wm_loss = None
            result = dict(policy_loss=None, value_loss=None, approx_kl=None, actor_updates=0, critic_updates=0, kl_stopped=False)
            if cfg.method == "ppo_sim":
                result = experiment.optimize_simulator()
            else:
                if not experiment.frozen:
                    step_accumulator += cfg.collect_steps
                    collected = int(step_accumulator)
                    step_accumulator -= collected
                    experiment.collect_model_data(collected)
                    wm_loss = experiment.fit_world_model()
                if cfg.method != "pretrain" and (cfg.bootstrap or experiment.frozen or i > cfg.wm_only_iters):
                    result = experiment.optimize_imagined()
            completed = i
            row = dict(iteration=i, method=cfg.method, seed=cfg.seed, lr=lr, wm_loss=wm_loss, **result, **ledger.snapshot())
            append_csv(output / "train.csv", row)
            print(json.dumps(row, allow_nan=False), flush=True)
            if i % cfg.checkpoint_every == 0:
                experiment.save(output / f"ckpt_{i:06d}.pt", i, manifest)
        experiment.save(output / f"ckpt_{completed:06d}.pt", completed, manifest)
        if cfg.method == "pretrain":
            experiment.save(output / "bootstrap.pt", completed, manifest, include_replay=True)
        write_json(output / "completion.json", dict(status=status, iterations=completed, ledger=ledger.snapshot()))


if __name__ == "__main__":
    main()
