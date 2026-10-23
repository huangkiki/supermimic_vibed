"""Recover missing jobs after a scheduler stop; never rerun started training or failed jobs.

The execution checkout and original budget remain fixed. This separate
orchestrator permits a resource-guard fix without changing evaluation sources.
Training that never started requires an explicit option and its original config.
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


def original_config(root, execution_root, args, method, seed):
    config = json.loads((execution_root / "experiments/ablation/configs" / f"{args['phase']}-{method}.json").read_text())
    config.update(seed=seed, subset=args["subset"])
    if method != "pretrain":
        for key in ("iterations", "checkpoint_every"):
            if args.get(key) is not None:
                config[key] = args[key]
    if method.startswith("ppo") and args.get("ppo_actor_lr") is not None:
        config["actor_lr"] = args["ppo_actor_lr"]
    if "_frozen" in method:
        config["bootstrap"] = str(root / f"pretrain-seed{seed}" / "bootstrap.pt")
    return config


def recovery_plan(root, execution_root, include_unstarted_training=False):
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
    interpreters = {r["command"][0] for r in commands if "experiments.ablation.train" in r["command"]}
    if len(interpreters) != 1:
        raise ValueError("A unique recorded training interpreter is required")
    interpreter = next(iter(interpreters))
    jobs = []
    for seed in args["seeds"]:
        for method in args.get("methods", METHOD_ORDER):
            run = root / f"{method}-seed{seed}"
            config_path = root / f"{method}-seed{seed}.json"
            training = [r for r in commands if "experiments.ablation.train" in r["command"]
                        and Path(r["command"][r["command"].index("--output") + 1]) == run]
            if not training:
                if not include_unstarted_training:
                    raise ValueError("All selected training runs must be complete before evaluation recovery")
                log = root / f"{method}-seed{seed}-train.log"
                if run.exists() or log.exists() or log.with_suffix(".gpu-memory.csv").exists():
                    raise ValueError("Unaccounted training artifacts exist; never restart a previously started run")
                config = original_config(root, execution_root, args, method, seed)
                if config_path.exists() and json.loads(config_path.read_text()) != config:
                    raise ValueError("Unstarted training config differs from the original campaign")
                bootstrap_hash = None
                if "_frozen" in method:
                    bootstrap = Path(config["bootstrap"])
                    pretrain = json.loads((bootstrap.parent / "completion.json").read_text())
                    pretrain_config = original_config(root, execution_root, args, "pretrain", seed)
                    allocated = [r for r in commands if "experiments.ablation.train" in r["command"]
                                 and Path(r["command"][r["command"].index("--output") + 1]) == bootstrap.parent]
                    if (len(allocated) != 1 or pretrain["status"] != "completed"
                            or pretrain["iterations"] != pretrain_config["iterations"]):
                        raise ValueError("Frozen training requires completed original pretraining")
                    bootstrap_hash = file_hash(bootstrap)
                    for manifest_path in root.glob(f"*-seed{seed}/manifest.json"):
                        manifest = json.loads(manifest_path.read_text())
                        if manifest.get("bootstrap_sha256") not in (None, bootstrap_hash):
                            raise ValueError("Bootstrap differs from the already trained frozen conditions")
                jobs.append(dict(command=[interpreter, "-m", "experiments.ablation.train", "--config", str(config_path),
                                          "--output", str(run)], log=log, cache=run / "cache",
                                 training_config=config, config_path=config_path, bootstrap_sha256=bootstrap_hash))
            else:
                if len(training) != 1:
                    raise ValueError("Missing unique successful training allocation")
                config = json.loads(config_path.read_text())
                if include_unstarted_training and config != original_config(root, execution_root, args, method, seed):
                    raise ValueError("Recorded training config differs from the original campaign")
                completion = json.loads((run / "completion.json").read_text())
                if completion["status"] != "completed" or completion["iterations"] != config["iterations"]:
                    raise ValueError("Previously started training must be complete; it cannot be restarted")
            if method == "pretrain":
                continue
            expected = {0, config["iterations"], *range(config["checkpoint_every"], config["iterations"], config["checkpoint_every"])}
            checkpoints = [run / f"ckpt_{iteration:06d}.pt" for iteration in sorted(expected)]
            if training and set(run.glob("ckpt_*.pt")) != set(checkpoints):
                raise ValueError("Missing or unexpected checkpoint files")
            for checkpoint in checkpoints:
                digest = file_hash(checkpoint) if training else None
                for repeat in range(repeats):
                    dest = run / checkpoint.stem
                    if repeats > 1:
                        dest /= f"repeat_{repeat:03d}"
                    command = [interpreter, "-m", "experiments.ablation.evaluate",
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
                    if dest.exists() or log.exists() or log.with_suffix(".gpu-memory.csv").exists():
                        raise ValueError("Unaccounted evaluation artifacts exist; preserve and inspect them")
                    jobs.append(dict(command=command, log=log, cache=run / "eval-cache",
                                     checkpoint=checkpoint, checkpoint_sha256=digest, expected_iterations=config["iterations"]))
    return campaign, jobs


def prepare_job(job):
    """Recheck immutable inputs immediately before launching one new subprocess."""
    if "training_config" in job:
        if job["cache"].parent.exists():
            raise ValueError("Training output appeared after recovery planning")
        config = job["training_config"]
        if job["bootstrap_sha256"] and file_hash(config["bootstrap"]) != job["bootstrap_sha256"]:
            raise ValueError("Bootstrap changed during recovery")
        path = job["config_path"]
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError("Training config changed during recovery")
        else:
            with path.open("x") as stream:
                stream.write(json.dumps(config, indent=2) + "\n")
    else:
        checkpoint = job["checkpoint"]
        completion = json.loads((checkpoint.parent / "completion.json").read_text())
        if completion["status"] != "completed" or completion["iterations"] != job["expected_iterations"]:
            raise ValueError("Training did not complete; evaluation cannot start")
        if job["checkpoint_sha256"] and file_hash(checkpoint) != job["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during recovery")
        if not checkpoint.is_file():
            raise ValueError("Expected checkpoint was not produced")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-unstarted-training", action="store_true",
                        help="Also launch original training stages with no previous allocation or output artifacts")
    args = parser.parse_args()
    root, execution_root = args.campaign.resolve(), args.execution_root.resolve()
    campaign, jobs = recovery_plan(root, execution_root, args.include_unstarted_training)
    training_count = sum("training_config" in job for job in jobs)
    print(json.dumps(dict(unstarted_training=training_count, missing_evaluations=len(jobs) - training_count,
                         remaining_gpu_hours=campaign["args"]["max_gpu_hours"] - campaign["actual_gpu_hours"])), flush=True)
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
            original_spend_gpu_hours=campaign["actual_gpu_hours"], missing_evaluations=len(jobs) - training_count,
            unstarted_training=training_count,
            scope="Missing jobs only; unchanged execution source/config and original total budget; never restart training"))
        campaign["status"] = "recovering_jobs" if training_count else "recovering_evaluations"
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
            prepare_job(job)
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
                raise RuntimeError("Recovery job failed; preserved evidence, no automatic retry")
            if "training_config" in job:
                completion = json.loads((job["cache"].parent / "completion.json").read_text())
                if completion["status"] != "completed" or completion["iterations"] != job["training_config"]["iterations"]:
                    raise RuntimeError("Recovered training hit its own limit; preserved evidence, no automatic retry")
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
