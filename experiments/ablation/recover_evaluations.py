"""Finish missing evaluations after a scheduler stop; never rerun training or failed jobs.

The execution checkout and original budget remain fixed. This separate
orchestrator permits a resource-guard fix without changing evaluation sources.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .run_suite import METHOD_ORDER, gpu_resources, source_snapshot, wait_for_idle_gpu


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_plan(root, execution_root):
    root, execution_root = Path(root).resolve(), Path(execution_root).resolve()
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign["status"] != "stopped":
        raise ValueError("Only a stopped campaign can be recovered")
    if not root.is_relative_to(execution_root) or source_snapshot(execution_root) != campaign["source_hashes"]:
        raise ValueError("Execution source/config changed; recovery must preserve the original checkout")
    commands = campaign["commands"]
    if any(r.get("returncode") != 0 or not math.isfinite(r["seconds"]) or r["seconds"] < 0 for r in commands):
        raise ValueError("Failed or unfinished subprocess requires diagnosis, not automatic recovery")
    if not math.isclose(sum(r["seconds"] for r in commands) / 3600, campaign["actual_gpu_hours"], abs_tol=1e-9):
        raise ValueError("Campaign allocation ledger is inconsistent")
    args = campaign["args"]
    if not math.isfinite(args["max_gpu_hours"]) or campaign["actual_gpu_hours"] >= args["max_gpu_hours"]:
        raise ValueError("Original finite GPU budget has no remaining allocation")
    completed = {tuple(r["command"]): r for r in commands}
    if len(completed) != len(commands):
        raise ValueError("Duplicate completed commands")
    repeats = args.get("evaluation_repeats", 1)
    jobs = []
    for seed in args["seeds"]:
        for method in args.get("methods", METHOD_ORDER):
            run = root / f"{method}-seed{seed}"
            config = json.loads((root / f"{method}-seed{seed}.json").read_text())
            completion = json.loads((run / "completion.json").read_text())
            if completion["status"] != "completed" or completion["iterations"] != config["iterations"]:
                raise ValueError("All selected training runs must be complete before evaluation recovery")
            training = [r for r in commands if "experiments.ablation.train" in r["command"]
                        and Path(r["command"][r["command"].index("--output") + 1]) == run]
            if len(training) != 1:
                raise ValueError("Missing unique successful training allocation")
            if method == "pretrain":
                continue
            expected = {0, config["iterations"], *range(config["checkpoint_every"], config["iterations"], config["checkpoint_every"])}
            checkpoints = sorted(run.glob("ckpt_*.pt"))
            if {int(p.stem.split("_")[-1]) for p in checkpoints} != expected:
                raise ValueError("Missing or unexpected checkpoint files")
            for checkpoint in checkpoints:
                digest = file_hash(checkpoint)
                for repeat in range(repeats):
                    dest = run / checkpoint.stem
                    if repeats > 1:
                        dest /= f"repeat_{repeat:03d}"
                    command = [training[0]["command"][0], "-m", "experiments.ablation.evaluate",
                               "--checkpoint", str(checkpoint), "--output", str(dest)]
                    if "evaluation_repeats" in args:
                        command += ["--repeat-index", str(repeat), "--repeat-count", str(repeats)]
                    suffix = f"-repeat{repeat}" if repeats > 1 else ""
                    log = root / f"{method}-seed{seed}-{checkpoint.stem}{suffix}-eval.log"
                    if tuple(command) in completed:
                        score = json.loads((dest / "evaluation.json").read_text())
                        if (score["checkpoint_sha256"] != digest or score.get("evaluation_repeat_index", 0) != repeat
                                or score.get("evaluation_repeat_count", 1) != repeats or not (dest / "per_clip.csv").is_file()):
                            raise ValueError("Completed evaluation evidence is inconsistent")
                        continue
                    if dest.exists() or log.exists():
                        raise ValueError("Unaccounted evaluation artifacts exist; preserve and inspect them")
                    jobs.append(dict(command=command, log=log, cache=run / "eval-cache"))
    return campaign, jobs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, execution_root = args.campaign.resolve(), args.execution_root.resolve()
    campaign, jobs = recovery_plan(root, execution_root)
    print(json.dumps(dict(missing_evaluations=len(jobs), remaining_gpu_hours=campaign["args"]["max_gpu_hours"] - campaign["actual_gpu_hours"])), flush=True)
    if args.dry_run:
        return
    lock = root / "evaluation-recovery.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    def persist():
        (root / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    try:
        index = len(campaign.get("recoveries", [])) + 1
        backup = root / f"campaign.before-evaluation-recovery-{index:03d}.json"
        with backup.open("x") as stream:
            stream.write((root / "campaign.json").read_text())
        campaign.setdefault("recoveries", []).append(dict(
            time=datetime.now(timezone.utc).isoformat(), backup=backup.name,
            orchestrator_sha256=file_hash(__file__), resource_guard_sha256=file_hash(Path(__file__).with_name("run_suite.py")),
            original_spend_gpu_hours=campaign["actual_gpu_hours"], missing_evaluations=len(jobs),
            scope="Missing evaluations only; unchanged execution source/config and original total budget"))
        campaign["status"] = "recovering_evaluations"
        campaign.pop("error", None)
        persist()
        gpu = campaign["args"]["gpu"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(execution_root),
                           OMP_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONUNBUFFERED="1")
        for job in jobs:
            if source_snapshot(execution_root) != campaign["source_hashes"]:
                raise RuntimeError("Execution source/config changed during recovery")
            remaining = (campaign["args"]["max_gpu_hours"] - campaign["actual_gpu_hours"]) * 3600
            if remaining <= 0:
                raise TimeoutError("Original campaign budget exhausted")
            started = time.perf_counter()
            resource, observations = wait_for_idle_gpu(gpu, campaign["resources"]["uuid"], min(30., remaining))
            forecast = campaign.get("forecast") or {}
            if float(resource["memory_free_mib"]) < max(12000., forecast.get("projected_peak_memory_mib", 0.) / .85):
                raise RuntimeError("GPU memory no longer satisfies original resource qualification")
            if shutil.disk_usage(root).free / 2**30 < forecast.get("minimum_disk_reserve_gib", 0.):
                raise RuntimeError("Original disk reserve exhausted")
            record = dict(command=job["command"], log=str(job["log"]), resources=resource,
                          preflight_observations=observations, recovery=index, peak_device_memory_used_mib=0.)
            campaign["commands"].append(record)
            environment.update(TORCHINDUCTOR_CACHE_DIR=str(job["cache"] / "inductor"), WARP_CACHE_PATH=str(job["cache"] / "warp"))
            persist()
            try:
                with job["log"].open("x") as stream, job["log"].with_suffix(".gpu-memory.csv").open("x", newline="") as memory:
                    writer = csv.writer(memory)
                    writer.writerow(["elapsed_seconds", "device_memory_used_mib", "utilization_percent"])
                    process = subprocess.Popen(job["command"], cwd=execution_root, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        while True:
                            elapsed = time.perf_counter() - started
                            if elapsed >= remaining:
                                raise subprocess.TimeoutExpired(job["command"], remaining)
                            current = gpu_resources(gpu)
                            used = float(current["memory_total_mib"]) - float(current["memory_free_mib"])
                            record["peak_device_memory_used_mib"] = max(record["peak_device_memory_used_mib"], used)
                            writer.writerow([elapsed, used, current["utilization_percent"]])
                            memory.flush()
                            try:
                                record["returncode"] = process.wait(timeout=min(1., remaining - elapsed))
                                break
                            except subprocess.TimeoutExpired:
                                continue
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        raise
            finally:
                record["seconds"] = time.perf_counter() - started
                campaign["actual_gpu_hours"] += record["seconds"] / 3600
                persist()
            if record["returncode"]:
                raise RuntimeError("Evaluation failed; preserved evidence, no automatic retry")
        campaign["status"] = "completed"
    except BaseException as error:
        campaign["status"] = "stopped"
        campaign["error"] = str(error)
        raise
    finally:
        persist()
        lock.unlink()


if __name__ == "__main__":
    main()
