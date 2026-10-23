"""Finish the registered seed sequence after an identified first-seed queue succeeds.

This orchestration runs outside the immutable training checkout. It never retries
training, increases budgets, or deletes artifacts to make a resource check pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .run_suite import METHOD_ORDER, campaign_scope, check_forecast, source_snapshot, wait_for_idle_gpu


def read_json(path):
    return json.loads(Path(path).read_text())


def process_identity(pid, proc_root=Path("/proc")):
    """Bind liveness to the process start time and argv, not a reusable PID alone."""
    path = proc_root / str(pid)
    stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] in ("Z", "X"):
        raise RuntimeError("Dependency process is terminal")
    argv = (path / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    return dict(pid=pid, start_ticks=int(stat[19]), argv=[a.decode() for a in argv])


def dependency_status(queue_path, identity, script, proc_root=Path("/proc")):
    """A partial status write is a transient observation only while the same process lives."""
    try:
        state = read_json(queue_path)
    except json.JSONDecodeError:
        state = {"status": "partial_status_write"}
    if state["status"] == "completed":
        return state
    if state["status"] not in ("waiting", "running", "partial_status_write"):
        raise RuntimeError(f"Predecessor stopped; preserve and inspect it: {state.get('error', state['status'])}")
    if identity is None:
        raise RuntimeError("A completed predecessor changed back to a nonterminal state")
    current = process_identity(identity["pid"], proc_root)
    if current != identity or str(script) not in current["argv"]:
        raise RuntimeError("Predecessor process identity changed")
    return state


def completed_cost(root, seed, plan, sources):
    """Check the terminal campaign and native acceptance receipt before allocating the next seed."""
    campaign = read_json(root / "campaign.json")
    args = campaign["args"]
    scope = campaign_scope(METHOD_ORDER, plan["iterations"], plan["checkpoint_every"],
                           plan["ppo_actor_lr"], plan["evaluation_repeats"])
    if (campaign["status"] != "completed" or args["seeds"] != [seed] or args["phase"] != "formal"
            or args["subset"] != plan["subset"] or campaign["scope"] != scope
            or campaign["source_hashes"] != sources or campaign["resources"]["uuid"] != plan["target_gpu_uuid"]
            or args["max_gpu_hours"] != plan["campaign_gpu_hours_cap_per_seed"]):
        raise ValueError("Predecessor campaign identity, scope or completion differs from the plan")
    commands = campaign["commands"]
    expected_evaluations = len(plan["methods"]) * len({0, plan["iterations"], *range(
        plan["checkpoint_every"], plan["iterations"], plan["checkpoint_every"])}) * plan["evaluation_repeats"]
    if (len(commands) != len(METHOD_ORDER) + expected_evaluations
            or any(c.get("returncode") != 0 or not math.isfinite(c["seconds"]) or c["seconds"] <= 0 for c in commands)):
        raise ValueError("Predecessor has missing, failed or unaccounted subprocesses")
    spend = sum(c["seconds"] for c in commands) / 3600
    if not math.isclose(spend, campaign["actual_gpu_hours"], abs_tol=1e-9) or spend > args["max_gpu_hours"]:
        raise ValueError("Predecessor cost ledger does not satisfy its original budget")
    validation = read_json(root / "validation.json")
    checks = validation["checks"]
    if (len(checks) != len(METHOD_ORDER) or {c["method"] for c in checks} != set(METHOD_ORDER)
            or any(c["seed"] != seed or not c["training_and_full_physical_evaluation_valid"] for c in checks)
            or validation["evaluation_processes"] != expected_evaluations
            or not math.isclose(validation["actual_gpu_hours"], spend, abs_tol=1e-9)):
        raise ValueError("Predecessor lacks complete native qualification")
    return spend


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--predecessor-queue", required=True, type=Path)
    parser.add_argument("--predecessor-pid-file", required=True, type=Path)
    parser.add_argument("--predecessor-script", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=42 * 3600,
                        help="Finite dependency wait; does not allocate a GPU or stop the predecessor on expiry")
    args = parser.parse_args()
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        raise ValueError("Dependency wait must have a finite positive bound")
    execution, output = args.execution_root.resolve(), args.output.resolve()
    plan, template = read_json(args.plan), read_json(args.forecast)
    seeds = plan["validation_seeds"]
    if (seeds != [1, 2, 3] or plan["development_seeds"] != [0]
            or set(plan["methods"]) != set(METHOD_ORDER) - {"pretrain"}
            or template["seeds"] != [seeds[0]]):
        raise ValueError("This continuation requires the registered first-seed forecast and seeds 1/2/3")
    cap, total_cap = plan["campaign_gpu_hours_cap_per_seed"], plan["campaign_gpu_hours_cap_total"]
    if not all(math.isfinite(v) and v > 0 for v in (cap, total_cap)) or cap * len(seeds) > total_cap:
        raise ValueError("Per-seed limits do not fit the registered total budget")
    qualification = read_json(execution / "logs/ablation/all40-smoke-v1/campaign.json")
    sources = qualification["source_hashes"]
    if qualification["status"] != "completed" or source_snapshot(execution) != sources:
        raise ValueError("Execution source/config changed since its native smoke qualification")
    campaign_paths = [execution / f"logs/ablation/confirmatory-{plan['subset']}-seed{seed}-v2" for seed in seeds]
    script = args.predecessor_script.resolve()
    try:
        already_complete = read_json(args.predecessor_queue)["status"] == "completed"
    except json.JSONDecodeError:
        already_complete = False
    if already_complete:
        completed_cost(campaign_paths[0], seeds[0], plan, sources)
        identity = None
    else:
        identity = process_identity(int(args.predecessor_pid_file.read_text()))
        if str(script) not in identity["argv"]:
            raise ValueError("Predecessor PID does not run the expected script")
    orchestrator = Path(__file__).resolve().parents[2]
    orchestrator_sources = source_snapshot(orchestrator)
    input_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (args.plan, args.forecast, script)}
    output.mkdir(parents=True, exist_ok=False)
    lock = execution / "logs/ablation/confirmatory-continuation.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    state = dict(status="waiting", pid=os.getpid(), created_at=datetime.now(timezone.utc).isoformat(), predecessor_identity=identity,
                 planned_seeds=seeds[1:], original_per_seed_gpu_hours=cap, original_total_gpu_hours=total_cap,
                 wait_limit_seconds=args.wait_seconds,
                 projected_remaining_gpu_hours=len(seeds[1:]) * template["projected_campaign_gpu_hours"],
                 execution_source_hashes=sources, plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                 orchestrator_source_hashes=orchestrator_sources, input_hashes=input_hashes,
                 orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 completed_seeds=[], preflights=[], commands=[])

    def persist():
        temporary = output / "queue.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "queue.json")

    environment = dict(os.environ, PYTHONPATH=str(execution), PYTHONUNBUFFERED="1")

    def execute(command, name, cwd=execution, env=environment):
        if (source_snapshot(execution) != sources or source_snapshot(orchestrator) != orchestrator_sources
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in input_hashes.items())):
            raise RuntimeError("Execution/orchestration source or registered inputs changed; no subprocess may start")
        record = dict(command=command, log=str(output / name), started_at=datetime.now(timezone.utc).isoformat())
        state["commands"].append(record)
        with (output / name).open("x") as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            persist()
            record["returncode"] = process.wait()
        persist()
        if record["returncode"]:
            raise RuntimeError(f"Subprocess failed; no automatic retry: {name}")

    persist()
    try:
        started = time.monotonic()
        while True:
            dependency = dependency_status(args.predecessor_queue, identity, script)
            state.update(predecessor_status=dependency["status"], wait_seconds=time.monotonic() - started,
                         checked_at=datetime.now(timezone.utc).isoformat())
            persist()
            if dependency["status"] == "completed":
                break
            if state["wait_seconds"] >= args.wait_seconds:
                raise TimeoutError("Dependency wait expired; predecessor was not stopped or restarted")
            time.sleep(min(30., args.wait_seconds - state["wait_seconds"]))
        for index, seed in enumerate(seeds[1:], start=1):
            prior_cost = sum(completed_cost(path, previous, plan, sources)
                             for previous, path in zip(seeds[:index], campaign_paths[:index]))
            state["completed_seeds"] = seeds[:index]
            if source_snapshot(execution) != sources:
                raise RuntimeError("Execution source/config changed while waiting")
            if campaign_paths[index].exists():
                raise FileExistsError("Future seed output already exists; preserve it instead of rerunning")
            forecast = {**template, "seeds": [seed]}
            resources, observations = wait_for_idle_gpu(args.gpu, plan["target_gpu_uuid"])
            free_disk = shutil.disk_usage(execution).free / 2**30
            check_forecast(forecast, resources, plan["subset"], [seed], min(cap, total_cap - prior_cost),
                           free_disk, campaign_scope(METHOD_ORDER, plan["iterations"], plan["checkpoint_every"],
                                                    plan["ppo_actor_lr"], plan["evaluation_repeats"]))
            if prior_cost + cap * (len(seeds) - index) > total_cap:
                raise RuntimeError("Reserved remaining seed caps exceed the original total budget")
            forecast_path = output / f"forecast-seed{seed}.json"
            with forecast_path.open("x") as stream:
                stream.write(json.dumps(forecast, indent=2) + "\n")
            state.update(status="running", active_seed=seed)
            state["preflights"].append(dict(seed=seed, resources=resources, idle_observations=observations,
                free_disk_gib=free_disk, previous_completed_gpu_hours=prior_cost, forecast=forecast))
            persist()
            execute([sys.executable, "-m", "experiments.ablation.run_suite", "--phase", "formal", "--iterations", str(plan["iterations"]),
                     "--checkpoint-every", str(plan["checkpoint_every"]), "--ppo-actor-lr", str(plan["ppo_actor_lr"]),
                     "--evaluation-repeats", str(plan["evaluation_repeats"]), "--seeds", str(seed), "--gpu", str(args.gpu),
                     "--max-gpu-hours", str(cap), "--forecast", str(forecast_path), "--output", str(campaign_paths[index])],
                    f"seed{seed}-suite.log")
            execute([sys.executable, "-m", "experiments.ablation.validate_run", str(campaign_paths[index])], f"seed{seed}-validation.log")
            completed_cost(campaign_paths[index], seed, plan, sources)
        execute([sys.executable, "-m", "experiments.ablation.analyze_cohort", "--plan", str(args.plan.resolve()),
                 "--campaigns", *map(str, campaign_paths), "--output", str(output / "cohort-analysis")],
                "cohort-analysis.log", cwd=orchestrator, env=dict(environment, PYTHONPATH=str(orchestrator)))
        state.update(status="completed", completed_seeds=seeds, active_seed=None)
    except BaseException as error:
        state.update(status="stopped", error=str(error))
        raise
    finally:
        persist()
        lock.unlink()


if __name__ == "__main__":
    main()
