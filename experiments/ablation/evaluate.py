"""One checkpoint per fresh native simulator process, full clips, no truncated proxy scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

STARTED = time.perf_counter()

import torch

from .common import Config, append_csv, identity, motion_files, sha256, write_json


@torch.no_grad()
def evaluate(env, pi, mdp, motion, cfg, record=None):
    from supermimic.mdp import MAX_STEP_REWARD
    from supermimic.rotations import mat_to_quat
    count = len(motion.names)
    ids = torch.arange(count, device="cuda:0")
    env.done_eval.zero_()
    env.failed_eval.zero_()
    env.reset(ids, motion.clip_start.clone())
    env.refresh()
    returns = torch.zeros(count, device="cuda:0")
    steps = torch.zeros(count, device="cuda:0", dtype=torch.long)
    history = None
    vector_steps = 0
    trajectory = {key: [] for key in ("root_p", "root_quat", "q", "f", "alive")}

    def record_state():
        state = env.state()
        values = dict(root_p=state[0], root_quat=mat_to_quat(state[1]), q=state[4],
                      f=env.f, alive=~env.done_eval)
        previous_alive = trajectory["alive"][-1] if trajectory["alive"] else None
        for key, value in values.items():
            snapshot = value.detach().cpu().clone()
            # Hold each exact terminal pose, including when render stride skips
            # that frame. Finished simulator slots may already have been reset.
            if previous_alive is not None:
                snapshot[~previous_alive] = trajectory[key][-1][~previous_alive]
            trajectory[key].append(snapshot)

    if record:
        record_state()
    while not bool(env.done_eval.all()):
        obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        history = obs[:, None].repeat(1, cfg.hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
        action = pi(history, motion.goal(env.f, cfg.goal_frames))
        alive = ~env.done_eval.clone()
        if not torch.isfinite(action[alive]).all():
            raise FloatingPointError("nonfinite evaluation action on a live episode")
        out = env.step(action)
        if not torch.isfinite(out["r"][alive]).all():
            raise FloatingPointError("nonfinite physical evaluation reward")
        returns += torch.where(alive, out["r"], 0.)
        steps += alive.long()
        vector_steps += 1
        # Save the actual terminal state BEFORE resetting finished slots. Renderers
        # freeze at the first alive=False frame, so a failure cannot look like a reset.
        if record:
            record_state()
        # Keep all slots finite and in bounds after they finish. Flags/score stay latched.
        done_ids = env.done_eval.nonzero().flatten()
        if len(done_ids):
            env.reset(done_ids, motion.clip_start[done_ids])
        env.f = torch.minimum(env.f, motion.clip_end - 1)
    norm = returns / ((motion.clip_len - 1) * MAX_STEP_REWARD)
    rows = [dict(motion=name, return_sum=float(returns[i]), normalized_return=float(norm[i]),
                 steps=int(steps[i]), reference_steps=int(motion.clip_len[i] - 1),
                 failed=bool(env.failed_eval[i])) for i, name in enumerate(motion.names)]
    if record:
        import numpy as np
        np.savez_compressed(record, names=np.array(motion.names),
                            **{key: torch.stack(values).numpy() for key, values in trajectory.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
                            ref_q=motion.q.cpu().numpy(), clip_start=motion.clip_start.cpu().numpy(),
                            clip_len=motion.clip_len.cpu().numpy())
    return dict(physical_return=float(norm.mean()),
                frame_weighted_return=float(returns.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)),
                tracking_failure_rate=float(env.failed_eval.float().mean()),
                failed_clips=int(env.failed_eval.sum()), total_clips=count,
                evaluation_simulator_steps=vector_steps * count, active_evaluation_steps=int(steps.sum()),
                per_clip=rows)


def main():
    started = STARTED
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--motion-dir")
    parser.add_argument("--record", action="store_true", help="Save trajectory.npz for the existing tools/render.py")
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1)
    add_launcher_args(parser)
    args = parser.parse_args()
    if not 0 <= args.repeat_index < args.repeat_count:
        raise ValueError("Evaluation repeat index must be within the declared repeat count")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("Use a new output directory for repeated evaluations")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    cfg = Config(**checkpoint["config"])
    if args.motion_dir:
        cfg.motion_dir = args.motion_dir
    files = motion_files(cfg)
    if {p.name: sha256(p) for p in files} != checkpoint["manifest"]["motions"]:
        raise ValueError("Evaluation motion identity differs from training")
    import supermimic.networks as networks
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.motion import MotionLib
    from supermimic.networks import Policy
    from supermimic.robot import default_joint_pos_list
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
    evaluation_manifest = identity(cfg, hash_bootstrap=False)
    for key in ("assets", "packages", "torch", "cuda"):
        if evaluation_manifest[key] != checkpoint["manifest"][key]:
            raise ValueError(f"Evaluation environment differs from training: {key}")
    for name, digest in evaluation_manifest["source_hashes"].items():
        if name.startswith("supermimic/") and digest != checkpoint["manifest"]["source_hashes"][name]:
            raise ValueError(f"Evaluation core source differs from training: {name}")
    kin = G1Kinematics("cuda:0")
    motion = MotionLib([str(p) for p in files], kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    pi = Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist).to("cuda:0")
    pi.load_state_dict(checkpoint["pi"])
    pi.eval()
    sim_cfg = make_sim("cuda:0", args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, len(files), motion, mdp, eval_mode=True)
        score = evaluate(env, pi, mdp, motion, cfg, output / "trajectory.npz" if args.record else None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for row in score.pop("per_clip"):
            append_csv(output / "per_clip.csv", row)
        result = dict(method=cfg.method, seed=cfg.seed, iteration=checkpoint["iteration"],
                      subset=cfg.subset, checkpoint_sha256=sha256(args.checkpoint),
                      manifest=checkpoint["manifest"], train_ledger=checkpoint["ledger"],
                      evaluation_manifest=evaluation_manifest,
                      evaluation_seconds=elapsed, evaluation_gpu_hours=elapsed / 3600,
                      evaluation_gpu=torch.cuda.get_device_name(0),
                      evaluation_gpu_uuid=evaluation_manifest["gpu_uuid"],
                      evaluation_repeat_index=args.repeat_index, evaluation_repeat_count=args.repeat_count,
                      trajectory_recorded=args.record,
                      protocol="native-full-clips-first-frame-fresh-process-v1", **score)
        write_json(output / "evaluation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
