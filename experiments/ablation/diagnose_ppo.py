"""Capture a native PPO batch once, then replay updates with fixed data and Adam state.

This is a diagnostic, never a policy-performance or throughput comparison.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
import sys

import torch

from .common import Config, Ledger, identity, motion_files, sha256, write_json
from .ppo import GaussianActor, ppo_update


def load_models(checkpoint, device):
    from supermimic.mdp import CRITIC_EXTRA, INV_DIM, OBS_DIM
    from supermimic.networks import Critic, Policy
    cfg = Config(**checkpoint["config"])
    actor = GaussianActor(Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist)).to(device)
    critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * cfg.goal_frames).to(device)
    actor.mean_policy.load_state_dict(checkpoint["pi"])
    with torch.no_grad():
        actor.log_std.copy_(checkpoint["log_std"])
    critic.load_state_dict(checkpoint["critic"])
    return actor, critic


@torch.no_grad()
def distribution_parameters(actor, observations, goals, chunk=2048):
    means, stds = [], []
    for start in range(0, len(observations), chunk):
        dist = actor.distribution(observations[start:start + chunk], goals[start:start + chunk])
        means.append(dist.mean)
        stds.append(dist.stddev)
    return torch.cat(means), torch.cat(stds)


@torch.no_grad()
def exact_kl(old, new):
    old_mean, old_std = old
    mean, std = new
    values = (torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square()) /
              (2 * std.square()) - .5).sum(-1)
    return dict(mean=float(values.mean()), p95=float(values.quantile(.95)), maximum=float(values.max()))


def capture(args):
    started = time.perf_counter()
    from isaaclab.app import launch_simulation
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot import default_joint_pos_list
    from .train import Experiment

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ck = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    if ck["config"]["method"] != "ppo_sim":
        raise ValueError("Capture currently requires a physical PPO-Sim checkpoint")
    cfg = replace(Config(**ck["config"]), num_envs=args.num_envs)
    torch.manual_seed(args.seed)
    kin = G1Kinematics("cuda:0")
    motion = MotionLib([str(p) for p in motion_files(cfg)], kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    manifest = identity(cfg)
    sim_cfg = make_sim("cuda:0", args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, cfg.num_envs, motion, mdp, gamma=cfg.gamma)
        experiment = Experiment(cfg, env, mdp, motion, Ledger(started), manifest)
        experiment.pi.load_state_dict(ck["pi"])
        with torch.no_grad():
            experiment.actor.log_std.copy_(ck["log_std"])
        experiment.critic.load_state_dict(ck["critic"])
        torch.manual_seed(args.seed)
        # No policy/norm update: collect one actual on-policy batch from the saved policy.
        batch = experiment.simulator_rollout()
        torch.save(dict(batch={k: v.cpu() for k, v in batch.items()}, config=asdict(cfg),
                        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
                        seed=args.seed, manifest=manifest), output / "batch.pt")
        torch.cuda.synchronize()
        write_json(output / "capture.json", dict(seconds=time.perf_counter() - started,
                   peak_torch_memory_bytes=torch.cuda.max_memory_allocated(), samples=cfg.num_envs * cfg.horizon,
                   checkpoint_sha256=sha256(args.checkpoint), source=manifest,
                   scope="Independent native diagnostic batch; not a physical evaluation or efficiency benchmark"))


def replay(args):
    started = time.perf_counter()
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        raise ValueError("learning rates must be finite and positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = torch.load(args.batch, map_location=args.device, weights_only=False)
    checkpoint_path = args.checkpoint or data["checkpoint"]
    if sha256(checkpoint_path) != data["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    ck = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    cfg = Config(**data["config"])
    batch = data["batch"]
    obs, goal = (batch[k].flatten(0, 1) for k in ("obs", "goal"))
    rows = []
    for lr in args.learning_rates:
        torch.manual_seed(data["seed"])
        actor, critic = load_models(ck, args.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, betas=(.9, .95))
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
        actor_optimizer.load_state_dict(copy.deepcopy(ck["opt_pi"]))
        critic_optimizer.load_state_dict(copy.deepcopy(ck["opt_v"]))
        for group in actor_optimizer.param_groups:
            group["lr"] = lr
        before = distribution_parameters(actor, obs, goal)
        with torch.no_grad():
            old_dist = torch.distributions.Normal(*before)
            log_difference = old_dist.log_prob(batch["action"].flatten(0, 1)).sum(-1) - batch["log_prob"].flatten()
            initial_approx_kl = float((log_difference.exp() - 1 - log_difference).mean())
        result = ppo_update(actor, critic, actor_optimizer, critic_optimizer, batch,
                            epochs=cfg.ppo_epochs, minibatches=cfg.minibatches, clip=cfg.clip,
                            entropy_coef=cfg.entropy_coef, target_kl=cfg.target_kl)
        after_update = distribution_parameters(actor, obs, goal)
        actor.mean_policy.obs_norm.update(batch["obs"][:, :, -1])
        actor.mean_policy.goal_norm.update(batch["goal"])
        after_norm = distribution_parameters(actor, obs, goal)
        rows.append(dict(learning_rate=lr, initial_approx_kl=initial_approx_kl,
                         max_initial_logprob_difference=float(log_difference.abs().max()), **result,
                         optimizer_exact_kl=exact_kl(before, after_update),
                         norm_only_exact_kl=exact_kl(after_update, after_norm),
                         total_exact_kl=exact_kl(before, after_norm)))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    source_names = ("experiments/ablation/diagnose_ppo.py", "experiments/ablation/ppo.py",
                    "experiments/ablation/common.py", "supermimic/networks.py")
    receipt = dict(checkpoint=str(Path(checkpoint_path).resolve()), checkpoint_sha256=data["checkpoint_sha256"],
                   batch_sha256=sha256(args.batch), samples=len(obs), seed=data["seed"],
                   replay_sources={name: sha256(root / name) for name in source_names},
                   target_kl=cfg.target_kl, seconds=time.perf_counter() - started, results=rows,
                   scope="Identical native rollout and optimizer state; exact Gaussian old-to-new KL, no learning conclusion")
    write_json(output / "replay.json", receipt)
    print(json.dumps(receipt, indent=2))
    if args.assert_max_kl is not None and any(r["total_exact_kl"]["mean"] > args.assert_max_kl for r in rows):
        raise SystemExit("FAIL: retained PPO update exceeds the declared mean KL threshold")


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    first = commands.add_parser("capture")
    first.add_argument("--checkpoint", required=True)
    first.add_argument("--output", required=True)
    first.add_argument("--num-envs", type=int, default=512)
    first.add_argument("--seed", type=int, default=9021)
    second = commands.add_parser("replay")
    second.add_argument("--batch", required=True)
    second.add_argument("--checkpoint", help="Relocated source checkpoint; SHA256 must match the captured batch")
    second.add_argument("--output", required=True)
    second.add_argument("--device", default="cuda:0")
    second.add_argument("--learning-rates", nargs="+", type=float, default=[2e-4])
    second.add_argument("--assert-max-kl", type=float)
    if "capture" in sys.argv:
        from isaaclab.app import add_launcher_args
        add_launcher_args(first)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    (capture if args.mode == "capture" else replay)(args)


if __name__ == "__main__":
    main()
