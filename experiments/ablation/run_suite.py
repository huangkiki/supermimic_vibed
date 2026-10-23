"""Serial, resource-checked, finite-budget campaign. Never launches unbounded training."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def gpu_resources(gpu):
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    fields = next(csv.reader([subprocess.check_output(command, text=True).strip()]))
    return dict(zip(("name", "uuid", "driver", "memory_total_mib", "memory_free_mib", "utilization_percent"), [x.strip() for x in fields]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-gpu-hours", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--subset", choices=("locomotion", "all40"), default="locomotion")
    parser.add_argument("--forecast", help="Required for formal runs: measured forecast JSON")
    args = parser.parse_args()
    if args.max_gpu_hours <= 0 or not args.max_gpu_hours < float("inf"):
        raise ValueError("Explicit finite positive campaign budget required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists():
        raise FileExistsError("Existing campaign: preserve it and select a new output directory")
    resources = gpu_resources(args.gpu)
    if float(resources["memory_free_mib"]) < 12000:
        raise RuntimeError("Less than 12 GiB available; resource qualification is required")
    if args.phase == "formal":
        if not args.forecast:
            raise ValueError("Formal runs require a measured forecast")
        forecast = json.loads(Path(args.forecast).read_text())
        if forecast["gpu"] != resources["name"] or forecast["subset"] != args.subset:
            raise ValueError("forecast GPU/subset mismatch")
        if forecast["projected_campaign_gpu_hours"] > args.max_gpu_hours:
            raise ValueError("forecast exceeds the campaign budget")
        if forecast["projected_peak_memory_mib"] > float(resources["memory_free_mib"]) * .85:
            raise ValueError("forecast leaves insufficient memory headroom")
    campaign = dict(args=vars(args), resources=resources, commands=[], actual_gpu_hours=0,
                    status="running", budget_scope="subprocess allocation time including setup, compilation and evaluation")
    def persist():
        (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    persist()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
                       TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONUNBUFFERED="1")

    def execute(command, log, cache):
        remaining = args.max_gpu_hours * 3600 - campaign["actual_gpu_hours"] * 3600
        if remaining <= 0:
            raise TimeoutError("Campaign GPU-hour budget exhausted")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        environment["WARP_CACHE_PATH"] = str(cache / "warp")
        start = time.perf_counter()
        record = dict(command=command, log=str(log), resources=gpu_resources(args.gpu))
        if float(record["resources"]["memory_free_mib"]) < 12000:
            raise RuntimeError("Available GPU memory changed; stopping before launch")
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
            for method in ("pretrain", "ppo_sim", "ppo_wm_online", "supermimic", "ppo_wm_frozen", "bptt_wm_frozen"):
                run = output / f"{method}-seed{seed}"
                raw = json.loads((configs / f"{args.phase}-{method}.json").read_text())
                raw.update(seed=seed, subset=args.subset)
                if method.endswith("_frozen"):
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
                    dest = run / checkpoint.stem
                    # A distinct cache for the evaluator gives every method the same cold-start convention.
                    execute([sys.executable, "-m", "experiments.ablation.evaluate", "--checkpoint", str(checkpoint), "--output", str(dest)],
                            output / f"{method}-seed{seed}-{checkpoint.stem}-eval.log", run / "eval-cache")
        campaign["status"] = "completed"
    except Exception as error:
        campaign["status"] = "stopped"
        campaign["error"] = str(error)
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
