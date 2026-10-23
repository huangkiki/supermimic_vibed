"""Serial, resource-checked, finite-budget campaign. Never launches unbounded training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import time


METHOD_ORDER = ("pretrain", "ppo_sim", "ppo_wm_online", "supermimic", "ppo_wm_frozen_hard", "ppo_wm_frozen", "bptt_wm_frozen")


def campaign_scope(methods, iterations=None, checkpoint_every=None, ppo_actor_lr=None, evaluation_repeats=1):
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHOD_ORDER):
        raise ValueError("Select unique known methods")
    if any("_frozen" in method for method in methods) and "pretrain" not in methods:
        raise ValueError("Frozen conditions require pretrain in the selected campaign")
    if any(value is not None and value < 1 for value in (iterations, checkpoint_every, evaluation_repeats)):
        raise ValueError("Iterations, checkpoint interval and evaluation repeats must be positive")
    if ppo_actor_lr is not None and (not math.isfinite(ppo_actor_lr) or ppo_actor_lr <= 0):
        raise ValueError("PPO actor learning rate must be finite and positive")
    return dict(methods=[method for method in METHOD_ORDER if method in methods], iterations=iterations,
                checkpoint_every=checkpoint_every, ppo_actor_lr=ppo_actor_lr, evaluation_repeats=evaluation_repeats)


def source_snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("supermimic", "experiments/ablation")
            for path in sorted((root / folder).rglob("*")) if path.suffix in (".py", ".json")}


def gpu_resources(gpu):
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    fields = next(csv.reader([subprocess.check_output(command, text=True).strip()]))
    return dict(zip(("name", "uuid", "driver", "memory_total_mib", "memory_free_mib", "utilization_percent"), [x.strip() for x in fields]))


def check_forecast(forecast, resources, subset, seeds, budget, free_disk_gib, scope=None):
    if forecast["gpu"] != resources["name"] or forecast["subset"] != subset:
        raise ValueError("forecast GPU/subset mismatch")
    if forecast["gpu_uuid"] != resources["uuid"] or forecast["seeds"] != seeds:
        raise ValueError("forecast GPU identity/seed matrix mismatch")
    if scope is not None and forecast.get("scope") != scope:
        raise ValueError("forecast method, tuning or evaluation scope mismatch")
    for key in ("projected_campaign_gpu_hours", "projected_peak_memory_mib", "projected_disk_gib", "minimum_disk_reserve_gib"):
        if not math.isfinite(forecast[key]) or forecast[key] <= 0:
            raise ValueError(f"Invalid forecast: {key}")
    if forecast["projected_campaign_gpu_hours"] > budget:
        raise ValueError("forecast exceeds the campaign budget")
    if forecast["projected_peak_memory_mib"] > float(resources["memory_free_mib"]) * .85:
        raise ValueError("forecast leaves insufficient memory headroom")
    if forecast["projected_disk_gib"] + forecast["minimum_disk_reserve_gib"] > free_disk_gib:
        raise ValueError("forecast leaves insufficient disk space")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-gpu-hours", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--subset", choices=("locomotion", "all40"), default="locomotion")
    parser.add_argument("--forecast", help="Required for formal runs: measured forecast JSON")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--iterations", type=int, help="Policy iterations only; pretrain retains its own budget")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--ppo-actor-lr", type=float, help="PPO conditions only; does not change BPTT/WM/Critic rates")
    parser.add_argument("--evaluation-repeats", type=int, default=1, help="Fresh processes per checkpoint, never training seeds")
    args = parser.parse_args()
    scope = campaign_scope(args.methods, args.iterations, args.checkpoint_every, args.ppo_actor_lr, args.evaluation_repeats)
    args.methods = scope["methods"]
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Training seeds must be unique and nonnegative")
    if args.max_gpu_hours <= 0 or not args.max_gpu_hours < float("inf"):
        raise ValueError("Explicit finite positive campaign budget required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists():
        raise FileExistsError("Existing campaign: preserve it and select a new output directory")
    resources = gpu_resources(args.gpu)
    forecast = None
    if float(resources["memory_free_mib"]) < 12000:
        raise RuntimeError("Less than 12 GiB available; resource qualification is required")
    if args.phase == "formal":
        if not args.forecast:
            raise ValueError("Formal runs require a measured forecast")
        forecast = json.loads(Path(args.forecast).read_text())
        check_forecast(forecast, resources, args.subset, args.seeds, args.max_gpu_hours,
                       shutil.disk_usage(output).free / 2**30,
                       scope if scope != campaign_scope(METHOD_ORDER) or "scope" in forecast else None)
    source_root = Path(__file__).resolve().parents[2]
    campaign = dict(args=vars(args), resources=resources, commands=[], actual_gpu_hours=0,
                    forecast=forecast, scope=scope,
                    source_hashes=source_snapshot(source_root),
                    status="running", budget_scope="subprocess allocation time including setup, compilation and evaluation")
    def persist():
        (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    persist()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
                       TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONUNBUFFERED="1")

    def execute(command, log, cache):
        if source_snapshot(source_root) != campaign["source_hashes"]:
            raise RuntimeError("Source/config changed during campaign; preserve this run and start a new campaign")
        remaining = args.max_gpu_hours * 3600 - campaign["actual_gpu_hours"] * 3600
        if remaining <= 0:
            raise TimeoutError("Campaign GPU-hour budget exhausted")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        environment["WARP_CACHE_PATH"] = str(cache / "warp")
        start = time.perf_counter()
        record = dict(command=command, log=str(log), resources=gpu_resources(args.gpu))
        if float(record["resources"]["memory_free_mib"]) < 12000:
            raise RuntimeError("Available GPU memory changed; stopping before launch")
        if forecast:
            current = record["resources"]
            if current["uuid"] != forecast["gpu_uuid"] or float(current["utilization_percent"]) > 5:
                raise RuntimeError("Qualified GPU is not idle or its identity changed")
            if forecast["projected_peak_memory_mib"] > float(current["memory_free_mib"]) * .85:
                raise RuntimeError("Available GPU memory no longer satisfies the measured forecast")
            if shutil.disk_usage(output).free / 2**30 < forecast["minimum_disk_reserve_gib"]:
                raise RuntimeError("Disk reserve exhausted; stopping before launch")
        campaign["commands"].append(record)
        persist()
        with log.open("w") as stream:
            try:
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record["peak_device_memory_used_mib"] = 0.
                record["memory_sampling_seconds"] = 1.
                try:
                    with log.with_suffix(".gpu-memory.csv").open("w", newline="") as memory_log:
                        writer = csv.writer(memory_log)
                        writer.writerow(["elapsed_seconds", "device_memory_used_mib", "utilization_percent"])
                        while True:
                            gpu = gpu_resources(args.gpu)
                            used = float(gpu["memory_total_mib"]) - float(gpu["memory_free_mib"])
                            record["peak_device_memory_used_mib"] = max(record["peak_device_memory_used_mib"], used)
                            elapsed = time.perf_counter() - start
                            writer.writerow([elapsed, used, gpu["utilization_percent"]])
                            memory_log.flush()
                            if elapsed >= remaining:
                                raise subprocess.TimeoutExpired(command, remaining)
                            try:
                                returncode = process.wait(timeout=min(1., remaining - elapsed))
                                break
                            except subprocess.TimeoutExpired:
                                if time.perf_counter() - start >= remaining:
                                    raise
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError("Campaign budget exhausted; stopped this run's process group")
                record["returncode"] = returncode
            finally:
                record["seconds"] = time.perf_counter() - start
                campaign["actual_gpu_hours"] += record["seconds"] / 3600
                persist()
        if returncode:
            raise RuntimeError(f"Command failed; preserved {log}")

    try:
        configs = Path(__file__).parent / "configs"
        for seed in args.seeds:
            for method in args.methods:
                run = output / f"{method}-seed{seed}"
                raw = json.loads((configs / f"{args.phase}-{method}.json").read_text())
                raw.update(seed=seed, subset=args.subset)
                if method != "pretrain":
                    for key in ("iterations", "checkpoint_every"):
                        if getattr(args, key) is not None:
                            raw[key] = getattr(args, key)
                if method.startswith("ppo") and args.ppo_actor_lr is not None:
                    raw["actor_lr"] = args.ppo_actor_lr
                if "_frozen" in method:
                    raw["bootstrap"] = str(output / f"pretrain-seed{seed}" / "bootstrap.pt")
                config_path = output / f"{method}-seed{seed}.json"
                config_path.write_text(json.dumps(raw, indent=2) + "\n")
                execute([sys.executable, "-m", "experiments.ablation.train", "--config", str(config_path), "--output", str(run)],
                        output / f"{method}-seed{seed}-train.log", run / "cache")
                completion = json.loads((run / "completion.json").read_text())
                if completion["status"] != "completed":
                    raise RuntimeError(f"{method}: per-run time budget reached; inspect timing before extending")
                if method == "pretrain":
                    continue
                for checkpoint in sorted(run.glob("ckpt_*.pt")):
                    for repeat in range(args.evaluation_repeats):
                        suffix = f"-repeat{repeat}" if args.evaluation_repeats > 1 else ""
                        dest = run / checkpoint.stem
                        if args.evaluation_repeats > 1:
                            dest /= f"repeat_{repeat:03d}"
                        # A distinct evaluation cache has the same cold-start convention for each method.
                        execute([sys.executable, "-m", "experiments.ablation.evaluate", "--checkpoint", str(checkpoint),
                                 "--output", str(dest), "--repeat-index", str(repeat), "--repeat-count", str(args.evaluation_repeats)],
                                output / f"{method}-seed{seed}-{checkpoint.stem}{suffix}-eval.log", run / "eval-cache")
        campaign["status"] = "completed"
    except Exception as error:
        campaign["status"] = "stopped"
        campaign["error"] = str(error)
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
