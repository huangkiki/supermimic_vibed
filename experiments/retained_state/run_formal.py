#!/usr/bin/env python3
"""Run the formal original-versus-retained-state SuperMimic comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2, 3, 4)
PHASES = {"screen": 4_000, "final": 14_000}
CONDITIONS = {
    "original": (),
    "retained": ("--wm_latent_dim", "128", "--wm_state_skip"),
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def training_command(
    python: str,
    *,
    run_dir: Path,
    motion_dir: Path,
    seed: int,
    iterations: int,
    condition: str,
) -> list[str]:
    command = [
        python,
        "supermimic/train.py",
        "--iterations", str(iterations),
        "--num_envs", "4096",
        "--buffer_steps", "1000",
        "--wm_batch", "2048",
        "--pi_batch", "4096",
        "--wm_only_iters", "500",
        "--wm_batches", "10",
        "--wm_window", "8",
        "--horizon", "32",
        "--utd", "0.4",
        "--ckpt_every", "2000",
        "--log_every", "100",
        "--seed", str(seed),
        "--motion_dir", str(motion_dir),
        "--log_dir", str(run_dir),
    ]
    command.extend(CONDITIONS[condition])
    return command


def evaluation_command(
    python: str,
    *,
    run_dir: Path,
    checkpoint: Path,
    motion_dir: Path,
) -> list[str]:
    return [
        python,
        "supermimic/evaluate.py",
        "--run_dir", str(run_dir),
        "--ckpt", str(checkpoint),
        "--motion_dir", str(motion_dir),
    ]


def require_idle_gpu(gpu: int) -> dict:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {}
    for line in output.splitlines():
        index, name, uuid, total, used, utilization = [
            field.strip() for field in line.split(",", 5)
        ]
        rows[int(index)] = {
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
        }
    if gpu not in rows:
        raise ValueError(f"GPU {gpu} is unavailable")
    resource = rows[gpu]
    if resource["memory_used_mib"] > 512 or resource["utilization_percent"] > 5:
        raise RuntimeError(f"GPU {gpu} is not idle: {resource}")
    return resource


def evaluate_gate(results: dict) -> dict:
    """Apply the pre-registered screen/final threshold to six completed runs."""
    missing = [
        f"{condition}-seed{seed}"
        for condition in CONDITIONS
        for seed in SEEDS
        if f"{condition}-seed{seed}" not in results
    ]
    if missing:
        return {"passed": False, "reason": "incomplete", "missing": missing}

    original = [results[f"original-seed{seed}"] for seed in SEEDS]
    retained = [results[f"retained-seed{seed}"] for seed in SEEDS]
    original_mean = sum(row["norm_return"] for row in original) / len(original)
    retained_mean = sum(row["norm_return"] for row in retained) / len(retained)
    relative_gain = (retained_mean - original_mean) / original_mean
    positive_seeds = sum(
        candidate["norm_return"] >= baseline["norm_return"]
        and candidate["n_failed"] <= baseline["n_failed"]
        for baseline, candidate in zip(original, retained)
    )
    original_failures = sum(row["n_failed"] for row in original)
    retained_failures = sum(row["n_failed"] for row in retained)
    passed = (
        relative_gain >= 0.03
        and retained_failures <= original_failures
        and positive_seeds >= 2
    )
    return {
        "passed": passed,
        "original_mean_return": original_mean,
        "retained_mean_return": retained_mean,
        "relative_return_gain": relative_gain,
        "original_failures_out_of_120": original_failures,
        "retained_failures_out_of_120": retained_failures,
        "positive_seeds": positive_seeds,
        "required_positive_seeds": 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--motion-dir", type=Path, default=ROOT / "data/lafan1_g1")
    parser.add_argument("--max-gpu-hours", type=float, default=6.0)
    parser.add_argument("--skip-gpu-check", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_gpu_hours <= 0:
        raise ValueError("--max-gpu-hours must be positive")
    if not args.motion_dir.is_dir():
        raise FileNotFoundError(args.motion_dir)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    resource = None if args.skip_gpu_check else require_idle_gpu(args.gpu)
    iterations = PHASES[args.phase]
    motion_files = {
        path.name: sha256(path)
        for path in sorted(args.motion_dir.glob("*.csv"))
    }
    if len(motion_files) != 40:
        raise ValueError(f"expected 40 LAFAN CSVs, found {len(motion_files)}")
    manifest = {
        "protocol_version": 1,
        "phase": args.phase,
        "iterations": iterations,
        "seeds": list(SEEDS),
        "conditions": {
            name: list(extra) for name, extra in CONDITIONS.items()
        },
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "motion_dir": str(args.motion_dir),
        "motion_files_sha256": motion_files,
        "gpu": resource,
        "max_gpu_hours": args.max_gpu_hours,
        "started_at": time.time(),
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    started = time.monotonic()
    results = {}

    try:
        for condition in CONDITIONS:
            for seed in SEEDS:
                used_hours = (time.monotonic() - started) / 3600
                if used_hours >= args.max_gpu_hours:
                    raise TimeoutError(
                        f"GPU-hour cap reached: {used_hours:.4f}/{args.max_gpu_hours:.4f}"
                    )
                run_id = f"{args.phase}-{condition}-seed{seed}"
                run_dir = output / "runs" / run_id
                if run_dir.exists():
                    raise FileExistsError(run_dir)
                run_dir.mkdir(parents=True)
                train = training_command(
                    args.python,
                    run_dir=run_dir,
                    motion_dir=args.motion_dir,
                    seed=seed,
                    iterations=iterations,
                    condition=condition,
                )
                with (run_dir / "stdout.log").open("w") as log:
                    subprocess.run(train, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                checkpoint = run_dir / f"ckpt_{iterations:06d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                evaluation = evaluation_command(
                    args.python,
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                    motion_dir=args.motion_dir,
                )
                with (run_dir / "evaluation.log").open("w") as log:
                    subprocess.run(evaluation, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                evaluation_json = checkpoint.with_name(
                    checkpoint.stem + "_eval.json"
                )
                if not evaluation_json.is_file():
                    raise FileNotFoundError(evaluation_json)
                metrics = json.loads(evaluation_json.read_text())
                results[f"{condition}-seed{seed}"] = {
                    "norm_return": float(metrics["norm_return"]),
                    "norm_return_framew": float(metrics["norm_return_framew"]),
                    "n_failed": int(metrics["n_failed"]),
                    "failed": list(metrics["failed"]),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
                write_json(output / "partial-results.json", results)
                manifest["elapsed_gpu_hours"] = (time.monotonic() - started) / 3600
                write_json(output / "manifest.json", manifest)
        results["gate"] = evaluate_gate({k: v for k, v in results.items() if k != "gate"})
        write_json(output / "results.json", results)
        manifest.update(
            status="complete",
            elapsed_gpu_hours=(time.monotonic() - started) / 3600,
            completed_at=time.time(),
        )
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            status="failed",
            error=repr(exc),
            elapsed_gpu_hours=(time.monotonic() - started) / 3600,
            completed_at=time.time(),
        )
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
