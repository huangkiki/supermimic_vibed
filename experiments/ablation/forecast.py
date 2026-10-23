"""Extrapolate bounded target-size measurements, without forecasting convergence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PHASES = ("collection_seconds", "wm_training_seconds", "imagined_rollout_seconds", "policy_optimization_seconds")


def calibrated_forecast(runs, *, seeds, gpu_uuid, iterations, checkpoint_every,
                        evaluation_repeats, evaluation_seconds, disk_gib_per_seed,
                        pretrain_iterations=500):
    """Use completed target-size runs, including PPO batches with full Actor updates.

    Probe devices may differ: this is a planning estimate, never a performance
    comparison. The actual campaign must qualify the declared target device.
    """
    from .run_suite import METHOD_ORDER, campaign_scope

    required = {"ppo_sim", "ppo_wm_online", "supermimic", "pretrain"}
    if set(runs) != required:
        raise ValueError("Calibrated forecasts require all three methods and pretrain")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds) or not gpu_uuid:
        raise ValueError("Unique nonnegative seeds and target GPU UUID are required")
    if any(not isinstance(value, int) or value < 1
           for value in (iterations, checkpoint_every, evaluation_repeats, pretrain_iterations)):
        raise ValueError("Positive integer iteration and evaluation counts required")
    for value in (evaluation_seconds, disk_gib_per_seed):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Finite positive evaluation and storage allowances required")
    measured, manifests = {}, {}
    for method, directory in runs.items():
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        cfg = manifest["config"]
        completion = json.loads((root / "completion.json").read_text())
        with (root / "train.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if completion["status"] != "completed" or not rows or int(rows[-1]["iteration"]) != completion["iterations"]:
            raise ValueError(f"Measurement must be completed and match its final CSV: {method}")
        if cfg["method"] != method:
            raise ValueError(f"Measurement method mismatch: {method}")
        if [int(row["iteration"]) for row in rows] != list(range(1, completion["iterations"] + 1)):
            raise ValueError("Training CSV has missing or duplicate iterations")
        campaign = json.loads((root.parent / "campaign.json").read_text())
        allocations = [r for r in campaign["commands"]
                       if "experiments.ablation.train" in r["command"]
                       and Path(r["command"][r["command"].index("--output") + 1]).name == root.name]
        if len(allocations) != 1 or allocations[0].get("returncode") != 0:
            raise ValueError(f"Missing successful process allocation: {method}")
        allocation = allocations[0]
        if not math.isfinite(allocation["seconds"]) or allocation["seconds"] <= 0:
            raise ValueError("Invalid process timing")
        if allocation["seconds"] < float(rows[-1]["train_wall_seconds"]):
            raise ValueError("Process allocation is shorter than its training ledger")
        peak = allocation["peak_device_memory_used_mib"]
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("Missing finite positive device memory measurement")
        recorded_uuid = allocation.get("resources", {}).get("uuid")
        manifest_uuid = manifest.get("gpu_uuid")
        if not (recorded_uuid or manifest_uuid) or (recorded_uuid and manifest_uuid and recorded_uuid != manifest_uuid):
            raise ValueError("Missing or conflicting measurement GPU identity")
        start = cfg["wm_only_iters"] if method == "ppo_wm_online" or method == "supermimic" else 0
        warm = [(a, b) for a, b in zip(rows, rows[1:]) if int(b["iteration"]) > start + 1][-30:]
        if len(warm) < 10:
            raise ValueError(f"At least ten warm policy intervals required: {method}")
        differences = {key: [float(b[key]) - float(a[key]) for a, b in warm]
                       for key in (*PHASES, "train_wall_seconds")}
        if any(not math.isfinite(v) or v < 0 for values in differences.values() for v in values):
            raise ValueError("Nonfinite or decreasing cumulative timing")
        upper = {key: max(values) for key, values in differences.items()}
        if upper["train_wall_seconds"] <= 0:
            raise ValueError("Nonpositive warm interval timing")
        full_updates = cfg["ppo_epochs"] * cfg["minibatches"]
        if method.startswith("ppo") and not any(int(b["actor_updates"]) == full_updates for _, b in warm):
            raise ValueError("Measurement never completed full Actor updates; cannot qualify their cost")
        if method != "pretrain" and iterations < completion["iterations"]:
            raise ValueError("Forecast target cannot be shorter than a measured training run")
        first_policy = next((float(b["train_wall_seconds"]) - float(a["train_wall_seconds"])
                             for a, b in zip(rows, rows[1:]) if int(b["iteration"]) == start + 1), 0.)
        target_iterations = pretrain_iterations if method == "pretrain" else iterations
        measured[method] = dict(observations=len(warm), measured_iterations=completion["iterations"],
                               gpu_uuid=manifest_uuid or recorded_uuid,
                               gpu_uuid_source="training_manifest" if manifest_uuid else "process_resource_record",
                               allocation_seconds=allocation["seconds"], upper_interval=upper,
                               cold_setup_compile_io_seconds=float(rows[0]["train_wall_seconds"]) + first_policy + 60.,
                               peak_device_memory_mib=allocation["peak_device_memory_used_mib"],
                               training_seconds=allocation["seconds"] + max(0, target_iterations - completion["iterations"]) * upper["train_wall_seconds"])
        manifests[method] = manifest
    common_config = ("subset", "num_envs", "pi_batch", "horizon", "hist", "goal_frames", "wm_window",
                     "wm_batch", "wm_batches", "collect_steps", "amp", "initial_std", "gamma", "lam",
                     "ppo_epochs", "minibatches", "warmup_steps", "buffer_steps", "lr")
    def identity(manifest):
        return {**{key: manifest[key] for key in ("gpu", "motions", "assets", "packages", "torch", "cuda")},
                "core_sources": {key: value for key, value in manifest["source_hashes"].items()
                                 if key.startswith("supermimic/")},
                "config": {key: manifest["config"][key] for key in common_config}}
    if len({json.dumps(identity(m), sort_keys=True) for m in manifests.values()}) != 1:
        raise ValueError("Forecast measurements differ in hardware model, physics, data or target-size configuration")
    actor_rates = {manifests[m]["config"].get("actor_lr") or manifests[m]["config"]["lr"]
                   for m in ("ppo_sim", "ppo_wm_online")}
    if len(actor_rates) != 1:
        raise ValueError("PPO Actor rates differ between calibrated measurements")
    actor_lr = next(iter(actor_rates))
    scope = campaign_scope(METHOD_ORDER, iterations, checkpoint_every, actor_lr, evaluation_repeats)
    pretrain = measured["pretrain"]["training_seconds"]
    frozen = {name: measured[source]["cold_setup_compile_io_seconds"] + iterations * sum(
              measured[source]["upper_interval"][key] for key in ("imagined_rollout_seconds", "policy_optimization_seconds"))
              for name, source in (("ppo_wm_frozen_hard", "ppo_wm_online"), ("ppo_wm_frozen", "ppo_wm_online"),
                                   ("bptt_wm_frozen", "supermimic"))}
    checkpoints = 1 + iterations // checkpoint_every + (iterations % checkpoint_every != 0)
    evaluations = len(seeds) * 6 * checkpoints * evaluation_repeats
    training = len(seeds) * (pretrain + sum(measured[m]["training_seconds"] for m in required - {"pretrain"})
                            + sum(frozen.values())) / 3600
    evaluation = evaluations * evaluation_seconds / 3600
    return dict(status="calibrated_planning_not_convergence", scope=scope, seeds=seeds,
                gpu=manifests["supermimic"]["gpu"], gpu_uuid=gpu_uuid,
                subset=manifests["supermimic"]["config"]["subset"], measurements=measured,
                measurement_gpu_uuids=sorted({m["gpu_uuid"] for m in measured.values()}),
                evaluation_processes=evaluations, assumed_evaluation_seconds_per_checkpoint=evaluation_seconds,
                projected_training_gpu_hours=training, projected_evaluation_gpu_hours=evaluation,
                projected_campaign_gpu_hours=(training + evaluation) * 1.25,
                pretrain_iterations=pretrain_iterations, pretrain_seconds_estimate=pretrain, frozen_training_seconds_estimates=frozen,
                projected_peak_memory_mib=1.2 * max(r["peak_device_memory_mib"] for r in measured.values()),
                projected_disk_gib=len(seeds) * disk_gib_per_seed, minimum_disk_reserve_gib=12.,
                sources={method: str(path) for method, path in runs.items()},
                limitations=["Frozen training throughput inferred from measured online policy phases",
                             "The actual campaign shares pretrain once; each standalone curve must charge it in full",
                             "Evaluation allowance and 25 percent time / 20 percent memory reserves are not confidence intervals",
                             "Mixed probe devices are disclosed for planning only; the actual comparison uses the target UUID",
                             "Finite training does not guarantee convergence; seed0 development data do not become independent validation"])


def intervals(path, count):
    rows = list(csv.DictReader(Path(path).open()))[-(count + 1):]
    return [{k: float(b[k]) - float(a[k]) for k in (*PHASES, "train_wall_seconds")}
            for a, b in zip(rows, rows[1:])]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cold-probe")
    parser.add_argument("--warm-bptt")
    parser.add_argument("--ppo-sim-run")
    parser.add_argument("--ppo-wm-run")
    parser.add_argument("--bptt-run")
    parser.add_argument("--pretrain-run")
    parser.add_argument("--gpu-uuid", help="Physical target device; calibrated probe devices may differ")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--evaluation-repeats", type=int, default=3)
    parser.add_argument("--evaluation-seconds", type=float, default=600.)
    parser.add_argument("--disk-gib-per-seed", type=float, default=12.)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    completed_runs = dict(ppo_sim=args.ppo_sim_run, ppo_wm_online=args.ppo_wm_run,
                          supermimic=args.bptt_run, pretrain=args.pretrain_run)
    if any(completed_runs.values()):
        if not all(completed_runs.values()) or not args.gpu_uuid:
            parser.error("All four completed runs and --gpu-uuid are required for calibration")
        result = calibrated_forecast(completed_runs, seeds=args.seeds, gpu_uuid=args.gpu_uuid,
                                     iterations=args.iterations, checkpoint_every=args.checkpoint_every,
                                     evaluation_repeats=args.evaluation_repeats,
                                     evaluation_seconds=args.evaluation_seconds, disk_gib_per_seed=args.disk_gib_per_seed,
                                     pretrain_iterations=json.loads((Path(__file__).parent / "configs/formal-pretrain.json").read_text())["iterations"])
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("seeds", "projected_campaign_gpu_hours", "projected_peak_memory_mib")}))
        return
    if not args.cold_probe or not args.warm_bptt:
        parser.error("Supply calibrated runs, or both legacy --cold-probe and --warm-bptt")
    cold, warm = Path(args.cold_probe), Path(args.warm_bptt)
    receipt = json.loads((cold / "resource-probe.json").read_text())
    runs = {r["method"]: r for r in receipt["runs"]}
    estimates = {}
    for method, run in runs.items():
        observations = intervals(warm / "run/train.csv", 10) if method == "supermimic" else intervals(cold / method / "train.csv", 2)
        upper = {k: max(r[k] for r in observations) for k in (*PHASES, "train_wall_seconds")}
        # Preserve measured cold setup/compile/IO cost; the warm run only supplies update rates.
        cold_rows = list(csv.DictReader((cold / method / "train.csv").open()))
        regular = sum(float(b["train_wall_seconds"]) - float(a["train_wall_seconds"])
                      for a, b in zip(cold_rows[1:], cold_rows[2:]))
        overhead = max(0., run["seconds"] - regular)
        model_only = upper["collection_seconds"] + upper["wm_training_seconds"]
        updates, pretraining = (4000, 0) if method == "ppo_sim" else (3500, 500)
        training = overhead + pretraining * model_only + updates * upper["train_wall_seconds"]
        estimates[method] = dict(observations=len(observations), upper_interval=upper,
                                 median_update_seconds=statistics.median(r["train_wall_seconds"] for r in observations),
                                 cold_setup_compile_io_seconds=overhead, training_seconds=training,
                                 peak_device_memory_mib=run["peak_device_memory_mib"])
    ppo, bptt = estimates["ppo_wm_online"], estimates["supermimic"]
    pretrain = ppo["cold_setup_compile_io_seconds"] + 500 * (ppo["upper_interval"]["collection_seconds"] + ppo["upper_interval"]["wm_training_seconds"]) + 60
    frozen = {name: source["cold_setup_compile_io_seconds"] + 4000 * (
              source["upper_interval"]["imagined_rollout_seconds"] + source["upper_interval"]["policy_optimization_seconds"])
              for name, source in (("ppo_wm_frozen_hard", ppo), ("ppo_wm_frozen", ppo), ("bptt_wm_frozen", bptt))}
    training = len(args.seeds) * (pretrain + sum(v["training_seconds"] for v in estimates.values()) + sum(frozen.values())) / 3600
    evaluations = len(args.seeds) * 6 * 5
    end_to_end = [training + evaluations * seconds / 3600 for seconds in (30, 300)]
    result = dict(status="planning_forecast_not_convergence", gpu=runs["supermimic"]["resources"]["name"],
                  gpu_uuid=runs["supermimic"]["resources"]["uuid"], subset="locomotion", seeds=args.seeds,
                  iterations=4000, measurements=estimates, pretrain_seconds_estimate=pretrain,
                  frozen_training_seconds_estimates=frozen, projected_training_gpu_hours=training,
                  assumed_evaluation_seconds_per_checkpoint=[30, 300],
                  projected_end_to_end_gpu_hours_range=end_to_end,
                  projected_campaign_gpu_hours=end_to_end[1] * 1.25,
                  projected_peak_memory_mib=max(r["peak_device_memory_mib"] for r in runs.values()) * 1.2,
                  projected_disk_gib=12 * len(args.seeds), minimum_disk_reserve_gib=12,
                  sources=dict(cold_probe=str(cold), warm_bptt=str(warm)),
                  limitations=["PPO rates have only two intervals; BPTT has ten warm intervals",
                               "Frozen throughput inferred from online phases, not separately measured at 4096",
                               "30-300 seconds per full-clip evaluation and 60 seconds bootstrap IO are planning allowances",
                               "25 percent time and 20 percent memory margins are engineering reserves, not confidence intervals",
                               "12 GiB per seed is an artifact allowance including full replay bootstrap, checkpoints and caches",
                               "Do not reuse this locomotion forecast for all40 without a separate preflight",
                               "4000 iterations do not guarantee meaningful tracking or convergence"])
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("seeds", "projected_end_to_end_gpu_hours_range", "projected_campaign_gpu_hours", "projected_peak_memory_mib")}))


if __name__ == "__main__":
    main()
