"""Fixed held-out trajectories, prediction errors, action-space directional derivatives.

This probes finite-horizon action gradients, not full policy-parameter gradients.
No critic bootstrap is used: the numerical simulator and WM objectives must match.
Both physical hard termination and the matching soft-survival diagnostic are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import Config, append_csv, freeze, motion_files, sha256, write_json
from .fidelity_math import agreement, contact_groups, reference_quality


def contact_signature(env):
    """Version-pinned MuJoCo-Warp collision records, NOT a foot-height proxy.

    Newton's MJWarp manager owns this solver. Fail on unsupported layouts instead
    of fabricating contacts. Sets contain active penetrating/touching geom pairs
    at the final physics substep; they do not measure contact force.
    """
    import warp as wp
    solver = env.sim.backend_manager._solver
    data = solver.mjw_data
    count = int(wp.to_torch(data.nacon).item())
    contact = data.contact
    world = wp.to_torch(contact.worldid)[:count].cpu().numpy()
    geom = wp.to_torch(contact.geom)[:count].cpu().numpy()
    dist = wp.to_torch(contact.dist)[:count].cpu().numpy()
    efc = wp.to_torch(contact.efc_address)[:count].cpu().numpy()
    if efc.ndim > 1:
        efc = efc[:, 0]
    signatures = [set() for _ in range(env.N)]
    for w, pair, distance, address in zip(world, geom, dist, efc):
        if 0 <= w < env.N and distance <= 0 and address >= 0:
            signatures[int(w)].add(tuple(sorted(int(x) for x in pair)))
    return [tuple(sorted(pairs)) for pairs in signatures]


@torch.no_grad()
def reset_recipe(env, bank):
    ids = torch.arange(env.N, device=env.device)
    env.reset(ids, bank["reset_frames"])
    env.done_eval.zero_()
    env.failed_eval.zero_()
    env.refresh()
    for action in bank["prefix_actions"]:
        env.step(action)
    env.done_eval.zero_()
    env.failed_eval.zero_()


def returns(rewards, done, gamma):
    survival = 1 - done
    alive = torch.cat((torch.ones_like(survival[:1]), survival.cumprod(0)[:-1]), 0)
    discounts = gamma ** torch.arange(len(rewards), device=rewards.device)[:, None]
    return (discounts * alive * rewards).sum(0)


@torch.no_grad()
def physical_rollout(env, bank, actions, gamma, contacts=False):
    from supermimic.mdp import flatten_state
    reset_recipe(env, bank)
    states, rewards, hard, soft, signatures = [flatten_state(env.state())], [], [], [], []
    for action in actions:
        out = env.step(action)
        states.append(out["s1"])
        rewards.append(out["r"])
        hard.append(out["term"].float())
        soft.append(env.mdp.soft_termination(env.state(), env.f))
        if contacts:
            signatures.append(contact_signature(env))
    state_tensor = torch.stack(states)
    reward_tensor = torch.stack(rewards)
    if not torch.isfinite(state_tensor).all() or not torch.isfinite(reward_tensor).all():
        raise FloatingPointError("nonfinite held-out physical rollout")
    return dict(states=state_tensor, hard_return=returns(reward_tensor, torch.stack(hard), gamma),
                soft_return=returns(reward_tensor, torch.stack(soft), gamma), contacts=signatures)


@torch.no_grad()
def collect_bank(env, pi, cfg, samples, horizon, prefix_steps, seed):
    """Independent reference starts + fixed reference-PD prefix, then collector-policy actions.

    This data is never inserted into a training replay buffer. It is held out as
    transitions, not as unseen motions: the motions remain the training library.
    """
    from supermimic.mdp import flatten_state
    generator = torch.Generator(device=env.device).manual_seed(seed)
    motion = env.motion
    clip = torch.arange(samples, device=env.device) % len(motion.names)
    available = motion.clip_len[clip] - prefix_steps - horizon - 5
    if (available <= 0).any():
        raise ValueError("held-out horizon exceeds clip length")
    frames = motion.clip_start[clip] + (torch.rand(samples, device=env.device, generator=generator) * available).long()
    bank = dict(reset_frames=frames, prefix_actions=[], probe_seed=seed, collector_seed=cfg.seed)
    reset_recipe(env, bank)
    prefix_contacts = []
    for _ in range(prefix_steps):
        action = (motion.q[env.f + 1] - env.default_q) / env.alpha
        bank["prefix_actions"].append(action.clone())
        env.step(action)
        prefix_contacts.append(contact_signature(env))
    bank["prefix_actions"] = torch.stack(bank["prefix_actions"])
    bank["state"] = flatten_state(env.state())
    bank["frames"] = env.f.clone()
    bank["xi"] = env.xi.clone()
    bank["previous_action"] = env.a_prev.clone()
    actions, history = [], None
    for _ in range(horizon):
        obs = env.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        history = obs[:, None].repeat(1, cfg.hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
        action = pi(history, motion.goal(env.f, cfg.goal_frames))
        actions.append(action)
        env.step(action)
    bank["actions"] = torch.stack(actions)
    bank["prefix_contacts"] = prefix_contacts
    bank["motion_names"] = [motion.names[int(i)] for i in clip]
    return bank


def imagined_rollout(wm, mdp, bank, actions, gamma):
    from supermimic.mdp import normalize_xi, unflatten_state
    state = unflatten_state(bank["state"])
    frames, previous = bank["frames"], bank["previous_action"]
    context = normalize_xi(bank["xi"])
    predictions, rewards, done = [], [], []
    for action in actions:
        state = wm.step(state, action, context)
        frames = frames + 1
        rewards.append(mdp.reward(state, frames, action, previous))
        done.append(mdp.soft_termination(state, frames))
        predictions.append(state)
        previous = action
    return returns(torch.stack(rewards), torch.stack(done), gamma), predictions


def probe_checkpoint(env, wm, bank, baseline, ck_hash, output, epsilons, directions, gamma, seed):
    from supermimic.mdp import unflatten_state
    actions = bank["actions"].detach().clone().requires_grad_(True)
    objective, predictions = imagined_rollout(wm, env.mdp, bank, actions, gamma)
    grad, = torch.autograd.grad(objective.sum(), actions)
    if not torch.isfinite(grad).all():
        raise FloatingPointError("nonfinite WM action gradient")
    samples = actions.shape[1]
    combined_contacts = bank["prefix_contacts"] + baseline["contacts"]
    labels = contact_groups(combined_contacts)
    first = len(bank["prefix_contacts"])
    initial_group = labels[first]
    group = ["switch_nearby" if any(row[n] == "switch_nearby" for row in labels[first:])
             else initial_group[n] for n in range(samples)]
    scale = torch.tensor([.3] * 3 + [.4] * 3 + [1.] * 3 + [3.14] * 3 + [1.] * 29 + [10.] * 29, device=env.device)
    segments = dict(position=(0, 3), orientation=(3, 6), linear_velocity=(6, 9), angular_velocity=(9, 12), joint_position=(12, 41), joint_velocity=(41, 70))
    for h in (1, 8, 16, 32):
        if h > len(predictions):
            continue
        error = wm.state_error(predictions[h - 1], unflatten_state(baseline["states"][h])).detach()
        for n in range(samples):
            row = dict(checkpoint_sha256=ck_hash, sample=n, motion=bank["motion_names"][n], horizon=h,
                       contact_group=group[n], initial_contact_group=initial_group[n],
                       scaled_mse=float((error[n] / scale).square().mean()))
            row.update({name + "_mse": float(error[n, a:b].square().mean()) for name, (a, b) in segments.items()})
            append_csv(output / "prediction.csv", row)
    repeat = physical_rollout(env, bank, actions.detach(), gamma)
    noise = {kind: (repeat[kind + "_return"] - baseline[kind + "_return"]).abs().cpu().numpy() for kind in ("hard", "soft")}
    initial_drift = (repeat["states"][0] - bank["state"]).abs().amax(-1)
    generator = torch.Generator(device=env.device).manual_seed(seed + 1)
    results = []
    for direction_id in range(directions + 1):
        direction = grad.detach().clone() if direction_id == directions else torch.randn(actions.shape, generator=generator, device=env.device)
        direction /= direction.square().sum((0, 2), keepdim=True).sqrt().clamp_min(1e-12)
        predicted = (grad * direction).sum((0, 2)).detach().cpu().numpy()
        measurements = []
        for eps in epsilons:
            positive = physical_rollout(env, bank, actions.detach() + eps * direction, gamma)
            negative = physical_rollout(env, bank, actions.detach() - eps * direction, gamma)
            measurements.append((eps, positive, negative))
        for kind in ("hard", "soft"):
            key = kind + "_return"
            derivatives = np.stack([((p[key] - m[key]) / (2 * eps)).cpu().numpy() for eps, p, m in measurements])
            deltas = np.stack([(p[key] - m[key]).cpu().numpy() for _, p, m in measurements])
            for n in range(samples):
                resolved = reference_quality(derivatives[:, n], deltas[:, n], noise[kind][n])
                if initial_drift[n] > 1e-5:
                    resolved[:] = False
                for j, (eps, plus, minus) in enumerate(measurements):
                    drift = max(float(initial_drift[n]),
                                float((plus["states"][0, n] - bank["state"][n]).abs().max()),
                                float((minus["states"][0, n] - bank["state"][n]).abs().max()))
                    resolved[j] = resolved[j] and drift <= 1e-5
                    row = dict(checkpoint_sha256=ck_hash, sample=n, motion=bank["motion_names"][n],
                               contact_group=group[n], initial_contact_group=initial_group[n], direction=direction_id,
                               direction_type="wm_gradient" if direction_id == directions else "random_unit_l2",
                               epsilon=eps, objective=kind, predicted_derivative=float(predicted[n]),
                               fd_derivative=float(derivatives[j, n]), resolved=bool(resolved[j]),
                               repeat_return_noise=float(noise[kind][n]), initial_state_max_drift=drift,
                               baseline_return=float(baseline[key][n]), positive_return=float(plus[key][n]),
                               negative_return=float(minus[key][n]), improvement=float(plus[key][n] - baseline[key][n]))
                    append_csv(output / "gradient.csv", row)
                    results.append(row)
    summaries = []
    for kind in ("hard", "soft"):
        for label in sorted(set(group)):
            # Each scale separately; don't count epsilon repeats as independent samples.
            for eps in epsilons:
                rows = [r for r in results if r["objective"] == kind and r["contact_group"] == label and r["epsilon"] == eps and r["direction_type"] == "random_unit_l2"]
                summaries.append(dict(objective=kind, contact_group=label, epsilon=eps,
                                      **agreement([r["fd_derivative"] for r in rows], [r["predicted_derivative"] for r in rows], [r["resolved"] for r in rows])))
    write_json(output / f"summary-{ck_hash[:12]}.json", dict(checkpoint_sha256=ck_hash, summaries=summaries,
        claim="finite-horizon action-space diagnostic; hard FD includes termination discontinuities; neither FD variant is an absolute Jacobian truth"))


def main():
    started = time.perf_counter()
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--collector", required=True, help="fixed policy for all model checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank", help="reuse an existing bank; never retrain on it")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--prefix-steps", type=int, default=16)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--probe-seed", type=int, default=9001)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.epsilons != sorted(set(args.epsilons)) or min(args.epsilons) <= 0 or len(args.epsilons) < 2:
        raise ValueError("At least two unique positive increasing perturbation scales are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prediction.csv").exists():
        raise FileExistsError("Use a new directory, preserve earlier probe evidence")
    import supermimic.networks as networks
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.motion import MotionLib
    from supermimic.networks import Policy, WorldModel
    from supermimic.robot import default_joint_pos_list
    torch.set_num_threads(4)
    # Full fp32 matmuls for the numerical diagnostic; training precision is recorded separately.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    networks.AMP_DTYPE = None
    torch.manual_seed(args.probe_seed)
    collector = torch.load(args.collector, map_location="cuda:0", weights_only=False)
    cfg = Config(**collector["config"])
    kin = G1Kinematics("cuda:0")
    files = motion_files(cfg)
    hashes = {p.name: sha256(p) for p in files}
    if hashes != collector["manifest"]["motions"]:
        raise ValueError("collector dataset differs")
    motion = MotionLib([str(p) for p in files], kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    pi = Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist).to("cuda:0")
    pi.load_state_dict(collector["pi"])
    sim_cfg = make_sim("cuda:0", args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.samples, motion, mdp, eval_mode=True)
        if args.bank:
            bank = torch.load(args.bank, map_location="cuda:0", weights_only=False)
            if bank["motion_hashes"] != hashes or bank["actions"].shape[1] != args.samples:
                raise ValueError("bank identity/batch mismatch")
            if bank["collector_sha256"] != sha256(args.collector) or bank["asset_hashes"] != collector["manifest"]["assets"]:
                raise ValueError("bank collector/asset mismatch")
        else:
            bank = collect_bank(env, pi, cfg, args.samples, 32, args.prefix_steps, args.probe_seed)
            bank.update(motion_hashes=hashes, collector_sha256=sha256(args.collector),
                        asset_hashes=collector["manifest"]["assets"],
                        held_out="independently generated transitions; same training motions; never fed to training")
            torch.save(bank, output / "heldout-bank.pt")
        baseline = physical_rollout(env, bank, bank["actions"], cfg.gamma, contacts=True)
        # The reset recipe's state is the authoritative FD initial state; retain initial drift.
        drift = float((baseline["states"][0] - bank["state"]).abs().max())
        if drift > 1e-5:
            raise RuntimeError(f"Initial state not reproducible by reset recipe: {drift}")
        torch.save(baseline, output / "physical-baseline.pt")
        for checkpoint in args.checkpoints:
            ck = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
            if ck["manifest"]["motions"] != hashes or ck["wm"] is None:
                raise ValueError("WM checkpoint dataset/architecture mismatch")
            if ck["manifest"]["assets"] != bank["asset_hashes"]:
                raise ValueError("WM checkpoint asset mismatch")
            wm = WorldModel().to("cuda:0")
            wm.load_state_dict(ck["wm"])
            freeze(wm, True)
            probe_checkpoint(env, wm, bank, baseline, sha256(checkpoint), output, args.epsilons,
                             args.directions, cfg.gamma, args.probe_seed)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        write_json(output / "probe.json", dict(seconds=elapsed, gpu_hours=elapsed / 3600, args=vars(args),
                   gpu=torch.cuda.get_device_name(0), collector_sha256=sha256(args.collector),
                   checkpoint_sha256={str(p): sha256(p) for p in args.checkpoints},
                   precision="fp32, TF32 disabled", initial_state_max_drift=drift))


if __name__ == "__main__":
    main()
