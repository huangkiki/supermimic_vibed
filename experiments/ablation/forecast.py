"""Extrapolate bounded target-size measurements, without forecasting convergence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


PHASES = ("collection_seconds", "wm_training_seconds", "imagined_rollout_seconds", "policy_optimization_seconds")

def intervals(path, count):
    rows = list(csv.DictReader(Path(path).open()))[-(count + 1):]
    return [{k: float(b[k]) - float(a[k]) for k in (*PHASES, "train_wall_seconds")}
            for a, b in zip(rows, rows[1:])]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cold-probe", required=True)
    parser.add_argument("--warm-bptt", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
