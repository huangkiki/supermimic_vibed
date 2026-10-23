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

STARTED = time.perf_counter()

import numpy as np
import torch

from .common import Config, append_csv, freeze, identity, motion_files, sha256, write_json
from .fidelity_math import agreement, body_contact_signatures, contact_groups, reference_quality, window_contact_groups


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
def restore_probe_state(env, bank):
    """Restore physical coordinates/velocities, with a fixed solver warm-start.

    Replaying a PD prefix has small numerical variation; it cannot guarantee an
    identical FD initial state. The bank prefix only constructs the held-out
    state and supplies contact history. Every numerical rollout restores that
    state directly through the same native write APIs used by TrackingSim.reset.
    """
    ids = torch.arange(env.N, device=env.device)
    if not torch.equal(env.xi, bank["xi"]):
        raise ValueError("Probe bank parameters differ from the nominal physical environment")
    env.reset(ids, bank["frames"])
    state = bank["state"]
    quaternion = state[:, 3:7]
    pose = torch.cat((state[:, :3] + env.scene.env_origins, quaternion[:, 1:], quaternion[:, :1]), -1)
    env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.robot.write_root_link_velocity_to_sim_index(root_velocity=state[:, 7:13], env_ids=ids)
    env.robot.write_joint_position_to_sim_index(position=state[:, 13:42], env_ids=ids)
    env.robot.write_joint_velocity_to_sim_index(velocity=state[:, 42:71], env_ids=ids)
    env.scene.reset(ids)
    env.a_prev.copy_(bank["previous_action"])
    env.refresh()
    # Pinned Newton reset(flags=0) clears persistent solver buffers without
    # overwriting the authored joint coordinates. This is a numerical convention.
    manager = env.sim.backend_manager
    manager._solver.reset(manager.backend.state_0, flags=0)
    env.done_eval.zero_()
    env.failed_eval.zero_()


def returns(rewards, done, gamma):
    survival = 1 - done
    alive = torch.cat((torch.ones_like(survival[:1]), survival.cumprod(0)[:-1]), 0)
    discounts = gamma ** torch.arange(len(rewards), device=rewards.device)[:, None]
    return (discounts * alive * rewards).sum(0)


def select_probe_reward(full_reward, terms, config, objective):
    """Use identical analytic terms on both sides, without subtractive cancellation."""
    from supermimic.motion import CONTROL_DT
    if objective == "full":
        return full_reward
    if objective == "state_only":
        return terms["track"] + config.w_joint_limit * terms["joint_limit"] * CONTROL_DT
    raise ValueError(f"Unknown probe objective: {objective}")


@torch.no_grad()
def physical_rollout(env, bank, actions, gamma, contacts=False, reward_objective="full"):
    from supermimic.mdp import flatten_state
    restore_probe_state(env, bank)
    states, rewards, hard, soft, signatures = [flatten_state(env.state())], [], [], [], []
    for action in actions:
        out = env.step(action)
        states.append(out["s1"])
        rewards.append(select_probe_reward(out["r"], out["terms"], env.mdp.cfg, reward_objective))
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
    def draw_frames():
        return motion.clip_start[clip] + (torch.rand(samples, device=env.device, generator=generator) * available).long()
    bank = dict(reset_frames=draw_frames(), probe_seed=seed, collector_seed=cfg.seed)
    attempts = torch.ones(samples, device=env.device, dtype=torch.long)
    # Keep the clip assignment fixed; reject starts whose reference-PD prefix
    # has already failed. Post-failure states are not an active control task.
    for _ in range(64):
        bank["prefix_actions"] = []
        env.reset(torch.arange(env.N, device=env.device), bank["reset_frames"])
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.refresh()
        prefix_contacts = []
        for _ in range(prefix_steps):
            action = (motion.q[env.f + 1] - env.default_q) / env.alpha
            bank["prefix_actions"].append(action.clone())
            env.step(action)
            prefix_contacts.append(contact_signature(env))
        if not torch.isfinite(flatten_state(env.state())).all():
            raise FloatingPointError("Nonfinite state during held-out bank collection")
        valid = ~env.failed_eval
        if bool(valid.all()):
            break
        attempts += (~valid).long()
        bank["reset_frames"] = torch.where(valid, bank["reset_frames"], draw_frames())
    else:
        raise RuntimeError("Could not collect live held-out starts for every clip within 64 attempts")
    bank["prefix_actions"] = torch.stack(bank["prefix_actions"])
    bank["start_sampling_attempts"] = attempts
    bank["prefix_rule"] = "fixed reference-PD prefix; no hard failure during prefix; per-clip rejection, maximum 64 attempts"
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


def imagined_rollout(wm, mdp, bank, actions, gamma, reward_objective="full"):
    from supermimic.mdp import normalize_xi, unflatten_state
    state = unflatten_state(bank["state"])
    frames, previous = bank["frames"], bank["previous_action"]
    context = normalize_xi(bank["xi"])
    predictions, rewards, done = [], [], []
    for action in actions:
        state = wm.step(state, action, context)
        frames = frames + 1
        reward, terms = mdp.reward(state, frames, action, previous, terms=True)
        rewards.append(select_probe_reward(reward, terms, mdp.cfg, reward_objective))
        done.append(mdp.soft_termination(state, frames))
        predictions.append(state)
        previous = action
    return returns(torch.stack(rewards), torch.stack(done), gamma), predictions


def probe_checkpoint(env, wm, bank, baseline, ck_hash, output, epsilons, directions, gamma, seed, gradient_horizon, reward_objective, contact_level, geom_body_ids):
    from supermimic.mdp import unflatten_state
    actions = bank["actions"][:gradient_horizon].detach().clone().requires_grad_(True)
    objective, predictions = imagined_rollout(wm, env.mdp, bank, actions, gamma, reward_objective)
    grad, = torch.autograd.grad(objective.sum(), actions)
    if not torch.isfinite(grad).all():
        raise FloatingPointError("nonfinite WM action gradient")
    samples = actions.shape[1]
    combined_contacts = bank["prefix_contacts"] + baseline["contacts"]
    geometry_labels = contact_groups(combined_contacts)
    labels = contact_groups(body_contact_signatures(combined_contacts, geom_body_ids)) if contact_level == "body" else geometry_labels
    first = len(bank["prefix_contacts"])
    initial_group = labels[first]
    group = window_contact_groups(labels, first, gradient_horizon)
    geometry_group = window_contact_groups(geometry_labels, first, gradient_horizon)
    if gradient_horizon != len(bank["actions"]):
        with torch.no_grad():
            _, predictions = imagined_rollout(wm, env.mdp, bank, bank["actions"], gamma, reward_objective)
    scale = torch.tensor([.3] * 3 + [.4] * 3 + [1.] * 3 + [3.14] * 3 + [1.] * 29 + [10.] * 29, device=env.device)
    segments = dict(position=(0, 3), orientation=(3, 6), linear_velocity=(6, 9), angular_velocity=(9, 12), joint_position=(12, 41), joint_velocity=(41, 70))
    for h in (1, 8, 16, 32):
        if h > len(predictions):
            continue
        error = wm.state_error(predictions[h - 1], unflatten_state(baseline["states"][h])).detach()
        prediction_groups = window_contact_groups(labels, first, h)
        geometry_prediction_groups = window_contact_groups(geometry_labels, first, h)
        for n in range(samples):
            row = dict(checkpoint_sha256=ck_hash, sample=n, motion=bank["motion_names"][n], horizon=h,
                       contact_group=prediction_groups[n], initial_contact_group=initial_group[n],
                       contact_group_level=contact_level, geometry_contact_group=geometry_prediction_groups[n],
                       scaled_mse=float((error[n] / scale).square().mean()))
            row.update({name + "_mse": float(error[n, a:b].square().mean()) for name, (a, b) in segments.items()})
            append_csv(output / "prediction.csv", row)
    if gradient_horizon != len(bank["actions"]):
        baseline = physical_rollout(env, bank, actions.detach(), gamma, reward_objective=reward_objective)
    repeat = physical_rollout(env, bank, actions.detach(), gamma, reward_objective=reward_objective)
    noise = {kind: (repeat[kind + "_return"] - baseline[kind + "_return"]).abs().cpu().numpy() for kind in ("hard", "soft")}
    initial_drift = (repeat["states"][0] - bank["state"]).abs().amax(-1)
    generator = torch.Generator(device=env.device).manual_seed(seed + 1)
    results = []
    for direction_id in range(directions + 1):
        direction = grad.detach().clone() if direction_id == directions else torch.randn(actions.shape, generator=generator, device=env.device)
        raw_norm = direction.square().sum((0, 2), keepdim=True).sqrt()
        direction /= raw_norm.clamp_min(1e-12)
        predicted = (grad * direction).sum((0, 2)).detach().cpu().numpy()
        measurements = []
        for eps in epsilons:
            positive = physical_rollout(env, bank, actions.detach() + eps * direction, gamma, reward_objective=reward_objective)
            negative = physical_rollout(env, bank, actions.detach() - eps * direction, gamma, reward_objective=reward_objective)
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
                               gradient_horizon=gradient_horizon,
                               reward_objective=reward_objective,
                               contact_group=group[n], initial_contact_group=initial_group[n], direction=direction_id,
                               contact_group_level=contact_level, geometry_contact_group=geometry_group[n],
                               direction_type="wm_gradient" if direction_id == directions else "random_unit_l2",
                               direction_l2_norm=float(direction[:, n].norm()),
                               wm_gradient_l2_norm=float(grad[:, n].norm()),
                               epsilon=eps, objective=kind, predicted_derivative=float(predicted[n]),
                               fd_derivative=float(derivatives[j, n]), resolved=bool(resolved[j]),
                               repeat_return_noise=float(noise[kind][n]), initial_state_max_drift=drift,
                               reproducible_initial_state=drift <= 1e-5,
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
    write_json(output / f"summary-{ck_hash[:12]}.json", dict(checkpoint_sha256=ck_hash, gradient_horizon=gradient_horizon,
        reward_objective=reward_objective, contact_group_level=contact_level, summaries=summaries,
        claim="finite-horizon action-space diagnostic; hard FD includes termination discontinuities; neither FD variant is an absolute Jacobian truth"))


def main():
    started = STARTED
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--collector", required=True, help="fixed policy for all model checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank", help="reuse an existing bank; never retrain on it")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--prefix-steps", type=int, default=16)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--gradient-horizon", type=int, choices=(1, 8, 16, 32), default=32,
                        help="Shorter fixed horizons can isolate stable-contact windows; analyze horizons separately")
    parser.add_argument("--reward-objective", choices=("full", "state_only"), default="full",
                        help="state_only removes the direct action-rate penalty on BOTH sides; it is a diagnostic, not the task score")
    parser.add_argument("--contact-level", choices=("body", "geometry"), default="body",
                        help="body merges changing contact patches on a single foot; raw geom-pair groups are always retained")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--probe-seed", type=int, default=9001)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.samples, args.directions, args.prefix_steps) < 1:
        raise ValueError("samples, directions and prefix-steps must be positive")
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
    probe_manifest = identity(cfg, hash_bootstrap=False)
    for key in ("assets", "packages", "torch", "cuda"):
        if probe_manifest[key] != collector["manifest"][key]:
            raise ValueError(f"Probe environment differs from collector: {key}")
    for name, digest in probe_manifest["source_hashes"].items():
        if name.startswith("supermimic/") and digest != collector["manifest"]["source_hashes"][name]:
            raise ValueError(f"Probe core source differs from collector: {name}")
    kin = G1Kinematics("cuda:0")
    files = motion_files(cfg)
    hashes = {p.name: sha256(p) for p in files}
    if hashes != collector["manifest"]["motions"]:
        raise ValueError("collector dataset differs")
    motion = MotionLib([str(p) for p in files], kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    pi = Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist).to("cuda:0")
    pi.load_state_dict(collector["pi"])
    pi.eval()
    sim_cfg = make_sim("cuda:0", args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.samples, motion, mdp, eval_mode=True)
        contact_model = env.sim.backend_manager._solver.mj_model
        geom_body_ids = [int(body) for body in contact_model.geom_bodyid]
        write_json(output / "contact-map.json", dict(geom_body_ids=geom_body_ids,
                   body_names=[contact_model.body(i).name for i in range(contact_model.nbody)]))
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
        baseline = physical_rollout(env, bank, bank["actions"], cfg.gamma, contacts=True, reward_objective=args.reward_objective)
        torch.save(baseline, output / "physical-baseline.pt")
        # Retain evidence even if the initial-state qualification fails.
        drift = float((baseline["states"][0] - bank["state"]).abs().max())
        if drift > 1e-5:
            raise RuntimeError(f"Initial state not reproducible by native state restore: {drift}")
        for checkpoint in args.checkpoints:
            ck = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
            if ck["manifest"]["motions"] != hashes or ck["wm"] is None:
                raise ValueError("WM checkpoint dataset/architecture mismatch")
            if ck["manifest"]["assets"] != bank["asset_hashes"]:
                raise ValueError("WM checkpoint asset mismatch")
            for key in ("packages", "torch", "cuda"):
                if ck["manifest"][key] != probe_manifest[key]:
                    raise ValueError(f"WM checkpoint environment mismatch: {key}")
            if ck["config"]["gamma"] != cfg.gamma:
                raise ValueError("WM checkpoint return objective mismatch")
            wm = WorldModel().to("cuda:0")
            wm.load_state_dict(ck["wm"])
            freeze(wm, True)
            probe_checkpoint(env, wm, bank, baseline, sha256(checkpoint), output, args.epsilons,
                             args.directions, cfg.gamma, args.probe_seed, args.gradient_horizon,
                             args.reward_objective, args.contact_level, geom_body_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        write_json(output / "probe.json", dict(seconds=elapsed, gpu_hours=elapsed / 3600, args=vars(args),
                   gpu=torch.cuda.get_device_name(0), collector_sha256=sha256(args.collector),
                   checkpoint_sha256={str(p): sha256(p) for p in args.checkpoints},
                   manifest=probe_manifest, peak_torch_memory_bytes=torch.cuda.max_memory_allocated(),
                   reset_protocol="native physical state restore, nominal xi, Newton solver reset flags=0 before each rollout",
                   precision="fp32, TF32 disabled", initial_state_max_drift=drift))


if __name__ == "__main__":
    main()
