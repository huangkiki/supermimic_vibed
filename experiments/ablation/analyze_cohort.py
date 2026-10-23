"""Analyze completed independent seed campaigns against a fixed, preregistered plan."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .analyze import collect_records, comparison_key, save_csv, write_analysis


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def validate_clip_summary(path, record):
    clips = read_csv(path.parent / "per_clip.csv")
    motions = record["manifest"]["motions"]
    if len(clips) != len(motions) or {r["motion"] + ".csv" for r in clips} != set(motions):
        raise ValueError("Physical per-clip coverage differs from the training motion set")
    values = [float(r["normalized_return"]) for r in clips]
    if not all(math.isfinite(v) for v in values) or any(r["failed"] not in ("True", "False") for r in clips):
        raise ValueError("Invalid per-clip scores or failure flags")
    if any(not 0 <= int(r["steps"]) <= int(r["reference_steps"]) or int(r["reference_steps"]) <= 0
           or (r["failed"] == "False" and int(r["steps"]) != int(r["reference_steps"])) for r in clips):
        raise ValueError("Successful evaluation ended before the full reference clip")
    failed = sum(r["failed"] == "True" for r in clips)
    if (record["total_clips"] != len(clips) or record["failed_clips"] != failed
            or not math.isclose(record["physical_return"], statistics.mean(values), abs_tol=2e-7, rel_tol=2e-6)
            or not math.isclose(record["tracking_failure_rate"], failed / len(clips), abs_tol=2e-7)):
        raise ValueError("Physical summary does not match its per-clip observations")


def collect_cohort(campaign_paths, plan):
    """Require the complete seed/method/checkpoint/repeat matrix and paid process ledger."""
    seeds = plan["validation_seeds"]
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            or set(seeds) & set(plan["development_seeds"])):
        raise ValueError("At least three distinct validation seeds, separate from development, are required")
    expected_methods = {"pretrain", *plan["methods"]}
    checkpoints = {0, plan["iterations"], *range(plan["checkpoint_every"], plan["iterations"], plan["checkpoint_every"])}
    rows, raw, receipts, seen_seeds = [], [], [], set()
    configs, source_snapshots = {}, set()
    for campaign_path in campaign_paths:
        path = Path(campaign_path).resolve()
        if path.is_dir():
            path /= "campaign.json"
        campaign, root = read_json(path), path.parent
        args = campaign["args"]
        actual_seeds = set(args["seeds"])
        if (not actual_seeds or len(actual_seeds) != len(args["seeds"]) or seen_seeds & actual_seeds
                or not actual_seeds <= set(seeds)):
            raise ValueError("Duplicate, development or undeclared training seed")
        seen_seeds.update(actual_seeds)
        if (campaign["status"] != "completed" or args["phase"] != "formal"
                or args["subset"] != plan["subset"] or set(args["methods"]) != expected_methods
                or len(args["methods"]) != len(expected_methods)
                or campaign["resources"]["uuid"] != plan["target_gpu_uuid"]):
            raise ValueError("Campaign completion, scope or physical GPU differs from the plan")
        for key in ("iterations", "checkpoint_every", "evaluation_repeats", "ppo_actor_lr"):
            if args.get(key) != plan[key]:
                raise ValueError(f"Campaign setting differs from the plan: {key}")
        commands = campaign["commands"]
        if any(c.get("returncode") != 0 or not math.isfinite(c["seconds"]) or c["seconds"] <= 0 for c in commands):
            raise ValueError("All allocated subprocesses must have complete successful cost records")
        spend = sum(c["seconds"] for c in commands) / 3600
        limits = (args["max_gpu_hours"], plan["campaign_gpu_hours_cap_per_seed"], plan["campaign_gpu_hours_cap_total"])
        if (not all(math.isfinite(value) and value > 0 for value in limits)
                or not math.isclose(spend, campaign["actual_gpu_hours"], abs_tol=1e-9)
                or spend > args["max_gpu_hours"]
                or args["max_gpu_hours"] > plan["campaign_gpu_hours_cap_per_seed"] * len(actual_seeds)):
            raise ValueError("Campaign cost ledger or registered budget is inconsistent")
        source_snapshots.add(json.dumps(campaign["source_hashes"], sort_keys=True))
        run_names = {f"{method}-seed{seed}" for method in expected_methods for seed in actual_seeds}
        if {p.parent.name for p in root.glob("*/manifest.json")} != run_names:
            raise ValueError("Training run coverage differs from the plan")
        qualification = read_json(root / "validation.json")
        checks = qualification["checks"]
        if (len(checks) != len(run_names)
                or {f"{r['method']}-seed{r['seed']}" for r in checks} != run_names
                or any(not r["training_and_full_physical_evaluation_valid"] for r in checks)
                or not math.isclose(qualification["actual_gpu_hours"], spend, abs_tol=1e-9)):
            raise ValueError("Native tensor/checkpoint qualification is incomplete or stale")
        paths, expected_commands = [], set()
        for seed in sorted(actual_seeds):
            bootstraps = set()
            for method in sorted(expected_methods):
                run = root / f"{method}-seed{seed}"
                manifest, completion = read_json(run / "manifest.json"), read_json(run / "completion.json")
                cfg = manifest["config"]
                if any(campaign["source_hashes"].get(name) != digest for name, digest in manifest["source_hashes"].items()):
                    raise ValueError("Training source identity differs from the campaign snapshot")
                iterations = plan["pretrain_iters"] if method == "pretrain" else plan["iterations"]
                if (cfg["method"] != method or cfg["seed"] != seed or cfg["subset"] != plan["subset"]
                        or cfg["iterations"] != iterations or completion["status"] != "completed"
                        or completion["iterations"] != iterations or manifest["gpu_uuid"] != plan["target_gpu_uuid"]):
                    raise ValueError("Training manifest/completion differs from the declared run")
                for key in ("num_envs", "horizon", "wm_window", "wm_only_iters", "initial_std"):
                    if cfg[key] != plan[key]:
                        raise ValueError(f"Training setting differs from the plan: {key}")
                actor_lr = plan["ppo_actor_lr"] if method.startswith("ppo") else plan["other_actor_wm_critic_lr"]
                if cfg["lr"] != plan["other_actor_wm_critic_lr"] or (cfg.get("actor_lr") or cfg["lr"]) != actor_lr:
                    raise ValueError("Actor/WM/Critic rates differ from the registered selection")
                if method != "pretrain" and cfg["checkpoint_every"] != plan["checkpoint_every"]:
                    raise ValueError("Checkpoint interval differs from the plan")
                signature = {k: v for k, v in cfg.items() if k not in ("seed", "bootstrap")}
                if method in configs and configs[method] != signature:
                    raise ValueError("Same method has different effective configuration across seeds")
                configs[method] = signature
                training = read_csv(run / "train.csv")
                if [int(r["iteration"]) for r in training] != list(range(1, iterations + 1)):
                    raise ValueError("Training CSV is incomplete or has duplicate iterations")
                expected_commands.add(("train", run.name))
                if "_frozen" in method:
                    if not manifest.get("bootstrap_sha256") or Path(cfg["bootstrap"]).parent.name != f"pretrain-seed{seed}":
                        raise ValueError("Frozen method has no same-seed bootstrap identity")
                    bootstraps.add(manifest["bootstrap_sha256"])
                if method == "pretrain":
                    continue
                for iteration in sorted(checkpoints):
                    for repeat in range(plan["evaluation_repeats"]):
                        dest = run / f"ckpt_{iteration:06d}"
                        if plan["evaluation_repeats"] > 1:
                            dest /= f"repeat_{repeat:03d}"
                        evaluation_path = dest / "evaluation.json"
                        record = read_json(evaluation_path)
                        if (record["method"] != method or record["seed"] != seed or record["iteration"] != iteration
                                or record["manifest"] != manifest
                                or record["evaluation_repeat_index"] != repeat
                                or record["evaluation_repeat_count"] != plan["evaluation_repeats"]):
                            raise ValueError("Evaluation does not identify its declared checkpoint/repeat")
                        validate_clip_summary(evaluation_path, record)
                        paths.append(evaluation_path)
                        raw.append(record)
                        expected_commands.add(("evaluate", run.name, iteration, repeat))
            if any("_frozen" in m for m in plan["methods"]) and len(bootstraps) != 1:
                raise ValueError("Frozen comparisons did not use the same bootstrap within each seed")
        if set(root.rglob("evaluation.json")) != set(paths):
            raise ValueError("Unexpected physical evaluations exist outside the declared matrix")
        allocated = []
        for c in commands:
            argv = c["command"]
            if "experiments.ablation.train" in argv:
                allocated.append(("train", Path(argv[argv.index("--output") + 1]).name))
            elif "experiments.ablation.evaluate" in argv:
                ck = Path(argv[argv.index("--checkpoint") + 1])
                allocated.append(("evaluate", ck.parent.name, int(ck.stem.split("_")[-1]),
                                  int(argv[argv.index("--repeat-index") + 1])))
            else:
                raise ValueError("Unrecognized subprocess in the campaign cost ledger")
        if len(allocated) != len(set(allocated)) or set(allocated) != expected_commands:
            raise ValueError("Missing or duplicate train/evaluation allocations")
        rows.extend(collect_records(paths, path))
        receipts.append(dict(campaign=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             seeds=sorted(actual_seeds), actual_gpu_hours=spend))
    if seen_seeds != set(seeds):
        raise ValueError(f"Missing validation seeds: {sorted(set(seeds) - seen_seeds)}")
    if len(source_snapshots) != 1 or len({comparison_key(r) for r in raw}) != 1:
        raise ValueError("Mixed source/data/GPU/physics/evaluation cohorts")
    if sum(r["actual_gpu_hours"] for r in receipts) > plan["campaign_gpu_hours_cap_total"]:
        raise ValueError("Total registered campaign budget exceeded")
    return rows, raw, receipts


def target_statistics(rows, plan):
    """First observed attainment; do not interpolate, or average only successful seeds."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], row["seed"]].append(row)
    expected = {(method, seed) for method in plan["methods"] for seed in plan["validation_seeds"]}
    if set(grouped) != expected:
        raise ValueError("Target statistics require the complete method/seed matrix")
    targets = [plan["primary_endpoint"]["return_threshold"], *plan["secondary_return_thresholds"]]
    failure_limit = plan["primary_endpoint"]["max_tracking_failure_rate"]
    observations, summary, comparisons = [], [], []
    axes = ("end_to_end_seconds", "total_gpu_hours", "simulator_steps", "total_simulator_steps")
    for threshold in targets:
        target_rows = {}
        for (method, seed), points in sorted(grouped.items()):
            points = sorted(points, key=lambda r: r["iteration"])
            attained = next((r for r in points if r["physical_return"] >= threshold
                             and r["tracking_failure_rate"] <= failure_limit), None)
            last = attained if attained is not None else points[-1]
            result = dict(method=method, seed=seed, return_threshold=threshold, max_failure_rate=failure_limit,
                          attained=attained is not None, observed_iteration=last["iteration"],
                          physical_return=last["physical_return"], tracking_failure_rate=last["tracking_failure_rate"],
                          cost_basis=last["cost_basis"], **{axis: last[axis] for axis in axes})
            observations.append(result)
            target_rows[method, seed] = result
        for method in plan["methods"]:
            members = [target_rows[method, seed] for seed in plan["validation_seeds"]]
            count = sum(r["attained"] for r in members)
            item = dict(method=method, return_threshold=threshold, attained=count, seeds=len(members),
                        censored=len(members) - count)
            for axis in axes:
                values = [r[axis] for r in members]
                item[axis + "_mean"] = statistics.mean(values) if count == len(members) else None
                item[axis + "_std"] = statistics.stdev(values) if count == len(members) else None
            summary.append(item)
        for baseline, candidate in (("ppo_sim", "ppo_wm_online"), ("ppo_wm_online", "supermimic"),
                                    ("ppo_wm_frozen", "bptt_wm_frozen")):
            if baseline not in plan["methods"] or candidate not in plan["methods"]:
                continue
            pairs = [(target_rows[baseline, seed], target_rows[candidate, seed]) for seed in plan["validation_seeds"]]
            count = sum(a["attained"] and b["attained"] for a, b in pairs)
            ratios = [a["total_gpu_hours"] / b["total_gpu_hours"] for a, b in pairs] if count == len(pairs) else []
            comparisons.append(dict(baseline=baseline, candidate=candidate, return_threshold=threshold,
                                    jointly_attained=count, seeds=len(pairs),
                                    paired_gpu_cost_ratio_mean=statistics.mean(ratios) if ratios else None,
                                    paired_gpu_cost_ratio_std=statistics.stdev(ratios) if ratios else None))
    return observations, summary, comparisons


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--campaigns", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = read_json(args.plan)
    rows, raw, receipts = collect_cohort(args.campaigns, plan)
    observations, summary, comparisons = target_statistics(rows, plan)
    write_analysis(rows, raw, args.output, f"{plan['subset']}: {len(plan['validation_seeds'])} independent training seeds")
    save_csv(args.output / "target-observations.csv", observations)
    save_csv(args.output / "target-summary.csv", summary)
    save_csv(args.output / "paired-target-costs.csv", comparisons)
    endpoints = [r for r in rows if r["iteration"] == plan["iterations"]]
    final = []
    for method in plan["methods"]:
        members = [r for r in endpoints if r["method"] == method]
        item = dict(method=method, seeds=len(members))
        for field in ("physical_return", "tracking_failure_rate", "total_gpu_hours", "simulator_steps", "imagined_steps"):
            item[field + "_mean"] = statistics.mean(r[field] for r in members)
            item[field + "_std"] = statistics.stdev(r[field] for r in members)
        final.append(item)
    save_csv(args.output / "final-seed-statistics.csv", final)
    receipt = dict(plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), plan=plan, campaigns=receipts,
                   actual_campaign_gpu_hours=sum(r["actual_gpu_hours"] for r in receipts),
                   standalone_method_gpu_hours=sum(r["total_gpu_hours"] for r in endpoints),
                   cost_semantics="Every frozen standalone method pays full pretrain; actual campaign spend counts it once",
                   target_semantics="First observed checkpoint; nonattainment is censored at the last observed cost; no interpolation",
                   time_semantics="Accounted training plus evaluation cost; not the chronological time at which offline scores became available",
                   ratio_semantics="Paired baseline/candidate GPU cost; only reported when all seeds attain both targets; residual method confounds remain")
    (args.output / "cohort.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
