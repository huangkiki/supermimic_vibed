"""Summarize gradient diagnostics using independent training seeds, never pooled states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .analyze import save_csv
from .analyze_cohort import collect_cohort, read_csv, read_json
from .compare_fidelity import compare
from .validate_fidelity import validate


CONTACT_GROUPS = ("all", "stable_contact", "switch_nearby", "flight")


def seed_statistics(rows, keys, fields, seeds):
    """Give each seed one vote; undefined metrics must not silently drop a seed."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct training seeds required")
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys)
        if row["seed"] not in seeds or row["seed"] in groups[key]:
            raise ValueError("Duplicate or undeclared seed observation")
        groups[key][row["seed"]] = row
    result = []
    for key, members in sorted(groups.items()):
        if set(members) != set(seeds):
            raise ValueError("Missing seed observation; do not average a partial cohort")
        item = dict(zip(keys, key), seeds=len(seeds))
        for field in fields:
            values = [members[s][field] for s in seeds if members[s][field] is not None]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite seed statistic")
            complete = len(values) == len(seeds)
            item[field + "_defined_seeds"] = len(values)
            item[field + "_mean"] = statistics.mean(values) if complete else None
            item[field + "_std"] = statistics.stdev(values) if complete else None
        result.append(item)
    return result


def pearson(x, y):
    if len(x) < 2 or any(v is None for v in [*x, *y]):
        return None
    if not np.isfinite([x, y]).all():
        raise ValueError("Nonfinite association")
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def microsteps(gradient, receipt, seed):
    """Pair gradient and random steps on reproducible states before averaging states."""
    args = receipt["args"]
    stages = {sha: int(Path(path).stem.split("_")[-1]) for path, sha in receipt["checkpoint_sha256"].items()}
    grouped = defaultdict(list)
    for row in gradient:
        direction = int(row["direction"])
        expected = "wm_gradient" if direction == args["directions"] else "random_unit_l2"
        if row["direction_type"] != expected or int(row["gradient_horizon"]) != args["gradient_horizon"]:
            raise ValueError("Direction type or gradient horizon differs from the recorded design")
        if row["contact_group"] not in CONTACT_GROUPS[1:]:
            raise ValueError("Unknown contact group")
        norm = float(row["direction_l2_norm"])
        if (any(float(row[k]) < 0 for k in ("repeat_return_noise", "initial_state_max_drift", "wm_gradient_l2_norm"))
                or (norm != 0 and not math.isclose(norm, 1., abs_tol=1e-5))
                or (norm == 0 and expected != "wm_gradient")):
            raise ValueError("Invalid numerical reference or perturbation normalization")
        grouped[row["checkpoint_sha256"], row["objective"], float(row["epsilon"])].append(row)
    result = []
    for (sha, objective, epsilon), rows in sorted(grouped.items()):
        states = defaultdict(list)
        for row in rows:
            states[int(row["sample"])].append(row)
        for group in CONTACT_GROUPS:
            selected = [rs for rs in states.values() if group == "all" or rs[0]["contact_group"] == group]
            paired = []
            for records in selected:
                if len({r["contact_group"] for r in records}) != 1:
                    raise ValueError("Contact grouping differs across directions")
                if (all(float(r["initial_state_max_drift"]) <= 1e-5 for r in records)
                        and all(float(r["direction_l2_norm"]) > 0 for r in records)):
                    grad = next(r for r in records if r["direction_type"] == "wm_gradient")
                    random = [r for r in records if r["direction_type"] == "random_unit_l2"]
                    if len(random) != args["directions"]:
                        raise ValueError("Incomplete per-state random control")
                    paired.append((float(grad["improvement"]), statistics.mean(float(r["improvement"]) for r in random),
                                   max(float(r["repeat_return_noise"]) for r in records)))
            means = [statistics.mean(values) for values in zip(*paired)] if paired else [None] * 3
            result.append(dict(seed=seed, iteration=stages[sha], gradient_horizon=args["gradient_horizon"],
                               reward_objective=args["reward_objective"], physical_objective=objective,
                               epsilon=epsilon, contact_group=group, unique_states=len(selected),
                               paired_states=len(paired), gradient_improvement=means[0], random_improvement=means[1],
                               gradient_minus_random=means[0] - means[1] if paired else None,
                               repeat_return_noise=means[2]))
    return result


def collect_seed(root, evaluations, training_manifest, plan):
    """Validate one paid six-probe campaign and link WM bytes to physical scores."""
    root = Path(root)
    campaign = read_json(root / "campaign.json")
    seed = campaign["plan"]["training_seed"]
    stages = {r["iteration"]: r for r in evaluations if r["seed"] == seed and r["method"] == plan["method"]}
    expected_hashes = {i: stages[i]["checkpoint_sha256"] for i in plan["wm_iterations"]}
    collector_hash = stages[plan["collector_iteration"]]["checkpoint_sha256"]
    expected = {f"gradient-components-{objective}-h{h}" for objective, h in
                itertools.product(plan["reward_objectives"], plan["gradient_horizons"])}
    if (campaign["status"] != "completed" or campaign["resources"]["uuid"] != plan["probe_gpu_uuid"]
            or {p.parent.name for p in root.glob("*/probe.json")} != expected):
        raise ValueError("Incomplete probe campaign or different GPU/design")
    commands = campaign["commands"]
    if any(c.get("returncode") != 0 or not math.isfinite(c["seconds"]) or c["seconds"] <= 0 for c in commands):
        raise ValueError("Missing successful probe allocation")
    allocations = []
    for command in commands:
        argv = command["command"]
        if "experiments.ablation.fidelity" not in argv:
            raise ValueError("Unexpected probe allocation")
        allocations.append(Path(argv[argv.index("--output") + 1]).name)
    cost = math.fsum(c["seconds"] for c in commands) / 3600
    if (len(allocations) != len(expected) or set(allocations) != expected
            or not math.isclose(cost, campaign["actual_gpu_hours"], abs_tol=1e-9)
            or not math.isfinite(campaign["plan"]["hard_gpu_hours"])
            or not 0 < cost <= campaign["plan"]["hard_gpu_hours"] <= plan["probe_gpu_hours_cap_per_seed"]):
        raise ValueError("Missing/duplicate probe allocations or inconsistent cost budget")
    receipts, improvements, banks, signatures = {}, [], set(), set()
    for name in sorted(expected):
        probe = root / name
        receipt = read_json(probe / "probe.json")
        args, manifest = receipt["args"], receipt["manifest"]
        if any(receipt[k] != plan[k] for k in ("precision", "reset_protocol")):
            raise ValueError("Probe precision or simulator reset protocol differs from registration")
        for field in ("samples", "prefix_steps", "prefix_controller", "directions", "epsilons", "contact_level", "probe_seed"):
            if args[field] != plan[field]:
                raise ValueError(f"Probe design differs from registration: {field}")
        if name != f"gradient-components-{args['reward_objective']}-h{args['gradient_horizon']}":
            raise ValueError("Probe folder does not identify its objective/horizon")
        hashes = {int(Path(path).stem.split("_")[-1]): sha for path, sha in receipt["checkpoint_sha256"].items()}
        if (hashes != expected_hashes or len(hashes) != len(receipt["checkpoint_sha256"])
                or receipt["collector_sha256"] != collector_hash):
            raise ValueError("WM or collector checkpoint bytes differ from physical evaluations")
        for field in ("config", "motions", "assets", "packages", "torch", "cuda", "gpu"):
            actual, expected_value = manifest[field], training_manifest[field]
            if field == "config":
                # Before independent PPO rates existed, omission meant the same
                # inherited rate as today's explicit actor_lr=None.
                actual = {"actor_lr": None, **actual}
                expected_value = {"actor_lr": None, **expected_value}
            if actual != expected_value:
                raise ValueError(f"Probe differs from its trained collector: {field}")
        core = lambda sources: {k: v for k, v in sources.items() if k.startswith("supermimic/")}
        if (manifest["gpu_uuid"] != plan["probe_gpu_uuid"]
                or core(manifest["source_hashes"]) != core(training_manifest["source_hashes"])
                or any(campaign["source_hashes"].get(k) != v for k, v in manifest["source_hashes"].items())):
            raise ValueError("Probe source or physical GPU identity differs")
        bank = read_json(probe / "bank.json")
        digest = receipt["bank_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef")
                or bank["collector_sha256"] != collector_hash or bank["samples"] != plan["samples"]):
            raise ValueError("Missing bank content or collector identity")
        banks.add(digest)
        verified = validate(probe, write_receipt=False)
        native = read_json(probe / "validation.json")
        if (native["bank_identity"] != "sha256_verified"
                or any(native.get(k) != v for k, v in verified.items() if k != "bank_identity")):
            raise ValueError("Native bank verification missing or probe recording receipt stale")
        signature = {k: receipt[k] for k in ("precision", "reset_protocol", "gpu")}
        signature["sources"] = manifest["source_hashes"]
        signatures.add(json.dumps(signature, sort_keys=True))
        receipts[name] = receipt
        improvements.extend(microsteps(read_csv(probe / "gradient.csv"), receipt, seed))
    if len(banks) != 1 or len(signatures) != 1:
        raise ValueError("All six probes must share one bank, implementation and reset protocol")
    paired = []
    for horizon in plan["gradient_horizons"]:
        rows = compare(root / f"gradient-components-full-h{horizon}", root / f"gradient-components-state_only-h{horizon}")
        indexed = {(r["iteration"], r["physical_objective"], r["epsilon"], r["contact_group"]): r for r in rows}
        for iteration, objective, epsilon, group in itertools.product(plan["wm_iterations"], ("hard", "soft"), plan["epsilons"], CONTACT_GROUPS):
            row = indexed.get((iteration, objective, epsilon, group))
            item = dict(seed=seed, iteration=iteration, gradient_horizon=horizon,
                        physical_objective=objective, epsilon=epsilon, contact_group=group)
            for field in ("total_directions", "unique_states", "paired_resolved", "full_own_resolved", "state_only_own_resolved"):
                item[field] = row[field] if row else 0
            for label, metric in itertools.product(("full", "state_only"), ("sign_agreement", "pearson")):
                item[f"{label}_{metric}"] = row[f"{label}_{metric}"] if row else None
            paired.append(item)
    prediction = read_csv(root / plan["canonical_prediction_probe"] / "prediction.csv")
    predictions = []
    for iteration, horizon in itertools.product(plan["wm_iterations"], plan["prediction_horizons"]):
        rows = [r for r in prediction if r["checkpoint_sha256"] == expected_hashes[iteration] and int(r["horizon"]) == horizon]
        score = stages[iteration]
        predictions.append(dict(seed=seed, iteration=iteration, prediction_horizon=horizon, states=len(rows),
                                scaled_mse=statistics.mean(float(r["scaled_mse"]) for r in rows),
                                physical_return=score["physical_return"], tracking_failure_rate=score["tracking_failure_rate"],
                                evaluation_repeats=score["evaluation_repeats"], evaluation_return_std=score["evaluation_return_std"]))
    identity = dict(seed=seed, bank_sha256=next(iter(banks)), collector_sha256=collector_hash,
                    signature=next(iter(signatures)), actual_gpu_hours=cost,
                    campaign_sha256=hashlib.sha256((root / "campaign.json").read_bytes()).hexdigest())
    return paired, improvements, predictions, identity


def stage_associations(paired, improvements, predictions, plan):
    result = []
    for seed, horizon in itertools.product(plan["validation_seeds"], plan["gradient_horizons"]):
        p = sorted((r for r in predictions if r["seed"] == seed and r["prediction_horizon"] == horizon), key=lambda r: r["iteration"])
        g = sorted((r for r in paired if r["seed"] == seed and r["gradient_horizon"] == horizon and r["contact_group"] == "all"
                    and r["physical_objective"] == "soft" and r["epsilon"] == plan["primary_epsilon"]), key=lambda r: r["iteration"])
        steps = sorted((r for r in improvements if r["seed"] == seed and r["gradient_horizon"] == horizon
                        and r["contact_group"] == "all" and r["physical_objective"] == "hard"
                        and r["reward_objective"] == "full" and r["epsilon"] == plan["primary_epsilon"]), key=lambda r: r["iteration"])
        if any([r["iteration"] for r in rows] != plan["wm_iterations"] for rows in (p, g, steps)):
            raise ValueError("Stage association requires all registered checkpoints")
        mse = [r["scaled_mse"] for r in p]
        result.append(dict(seed=seed, horizon=horizon, stages=len(p),
                           mse_return_pearson=pearson(mse, [r["physical_return"] for r in p]),
                           mse_sign_pearson=pearson(mse, [r["full_sign_agreement"] for r in g]),
                           mse_step_pearson=pearson(mse, [r["gradient_improvement"] for r in steps])))
    return result


def collect_fidelity_cohort(roots, physical_rows, physical_raw, plan):
    seeds = plan["validation_seeds"]
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds) or set(seeds) & set(plan["development_seeds"])
            or len(roots) != len(seeds) or plan["primary_epsilon"] not in plan["epsilons"]):
        raise ValueError("Complete independent validation seeds and registered perturbations required")
    if (plan["gradient_horizons"] != [1, 8, 32] or plan["prediction_horizons"] != [1, 8, 16, 32]
            or plan["reward_objectives"] != ["full", "state_only"]
            or plan["canonical_prediction_probe"] != "gradient-components-full-h32"
            or plan["wm_iterations"] != sorted(set(plan["wm_iterations"]))
            or len(plan["wm_iterations"]) < 2):
        raise ValueError("Expected registered stage/horizon/objective design and canonical prediction probe")
    if any(not math.isfinite(plan[k]) or plan[k] <= 0 for k in ("probe_gpu_hours_cap_per_seed", "probe_gpu_hours_cap_total")):
        raise ValueError("Probe cohort requires finite positive registered budgets")
    paired, improvements, predictions, receipts = [], [], [], []
    for root in roots:
        seed = read_json(Path(root) / "campaign.json")["plan"]["training_seed"]
        if seed not in seeds or seed in {r["seed"] for r in receipts}:
            raise ValueError("Duplicate or development probe seed")
        manifests = [r["manifest"] for r in physical_raw if r["seed"] == seed and r["method"] == plan["method"]]
        if not manifests or manifests[0]["config"]["subset"] != plan["subset"]:
            raise ValueError("Probe training cohort or motion subset missing")
        p, i, e, receipt = collect_seed(root, physical_rows, manifests[0], plan)
        paired.extend(p)
        improvements.extend(i)
        predictions.extend(e)
        receipts.append(receipt)
    if (len({r["signature"] for r in receipts}) != 1
            or any(len({r[field] for r in receipts}) != len(seeds) for field in ("bank_sha256", "collector_sha256"))):
        raise ValueError("Mixed probe implementation or reused bank/collector across training seeds")
    if not sum(r["actual_gpu_hours"] for r in receipts) <= plan["probe_gpu_hours_cap_total"]:
        raise ValueError("Registered total probe budget exceeded")
    return paired, improvements, predictions, receipts


def plot_cohort(tables, plan, output):
    """Plot seed SD, keeping contact coverage and physical improvement explicit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = max(plan["wm_iterations"])
    horizons = plan["gradient_horizons"]
    def errorbar(ax, rows, field, label, **style):
        rows = sorted((r for r in rows if r[field + "_mean"] is not None), key=lambda r: r["epsilon"])
        if rows:
            ax.errorbar([r["epsilon"] for r in rows], [r[field + "_mean"] for r in rows],
                        yerr=[r[field + "_std"] for r in rows], label=label, marker="o", capsize=3, **style)
        ax.set_xscale("log")
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(2, len(horizons), figsize=(12, 7), layout="constrained", squeeze=False)
    for col, horizon in enumerate(horizons):
        rows = [r for r in tables["gradient"] if r["iteration"] == final and r["gradient_horizon"] == horizon
                and r["physical_objective"] == "soft" and r["contact_group"] == "all"]
        for label in ("full", "state_only"):
            errorbar(axes[0, col], rows, label + "_sign_agreement", label)
            errorbar(axes[1, col], rows, label + "_pearson", label)
        axes[0, col].set(title=f"H={horizon}: matched soft objective", ylabel="Sign agreement")
        axes[1, col].set(xlabel="Action perturbation scale", ylabel="Pearson correlation")
    axes[0, 0].legend()
    fig.suptitle(f"{len(plan['validation_seeds'])} independent training seeds: mean ± sample SD\n"
                 "Paired resolved directions within each seed; undefined seed metrics remain blank")
    fig.savefig(output / "gradient-cohort.png", dpi=180)
    fig.savefig(output / "gradient-cohort.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, len(horizons), figsize=(12, 8), layout="constrained", squeeze=False)
    for col, horizon in enumerate(horizons):
        rows = [r for r in tables["microsteps"] if r["iteration"] == final and r["gradient_horizon"] == horizon
                and r["physical_objective"] == "hard" and r["reward_objective"] == "full" and r["contact_group"] == "all"]
        for field in ("gradient_improvement", "random_improvement"):
            errorbar(axes[0, col], rows, field, field.replace("_improvement", ""))
        axes[0, col].axhline(0, color="gray", linewidth=.8)
        axes[0, col].set(title=f"H={horizon}: physical microstep", ylabel="Physical return change", xlabel="Action step size")
        groups = CONTACT_GROUPS[1:]
        contact = {r["contact_group"]: r for r in tables["gradient"] if r["iteration"] == final
                   and r["gradient_horizon"] == horizon and r["physical_objective"] == "soft"
                   and r["epsilon"] == plan["primary_epsilon"] and r["contact_group"] != "all"}
        for i, group in enumerate(groups):
            row = contact[group]
            if row["full_sign_agreement_mean"] is not None:
                axes[1, col].errorbar(i, row["full_sign_agreement_mean"], yerr=row["full_sign_agreement_std"], fmt="o", capsize=3)
            label = f"defined {row['full_sign_agreement_defined_seeds']}/{row['seeds']} seeds\nmean states {row['unique_states_mean']:.1f}"
            axes[1, col].text(i, -.16, label, transform=axes[1, col].get_xaxis_transform(), ha="center", fontsize=7)
        axes[1, col].set(xticks=range(len(groups)), xticklabels=("stable", "switch", "flight"), xlim=(-.5, 2.5),
                         ylabel="Full-reward sign agreement", title=f"Contact groups, ε={plan['primary_epsilon']}")
        axes[1, col].grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Mean ± training-seed SD; sparse/absent contact groups retain coverage counts\n"
                 "Microsteps pair gradient/random directions on reproducible states; FD is a numerical reference")
    fig.savefig(output / "microsteps-contact-cohort.png", dpi=180)
    fig.savefig(output / "microsteps-contact-cohort.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, horizon in zip(axes.flat, plan["prediction_horizons"]):
        rows = sorted((r for r in tables["prediction-performance"] if r["prediction_horizon"] == horizon), key=lambda r: r["iteration"])
        for row in rows:
            ax.errorbar(row["scaled_mse_mean"], row["physical_return_mean"], xerr=row["scaled_mse_std"],
                        yerr=row["physical_return_std"], fmt="o", capsize=3, label=f"it {row['iteration']}")
        ax.set(xlabel=f"{horizon}-step state MSE", ylabel="Physical evaluation return")
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Prediction error and policy performance: mean ± training-seed SD\n"
                 "One canonical bank replay per seed; observational stage association, not causality")
    fig.savefig(output / "prediction-performance-cohort.png", dpi=180)
    fig.savefig(output / "prediction-performance-cohort.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--training-campaigns", nargs="+", type=Path, required=True)
    parser.add_argument("--probe-campaigns", nargs="+", type=Path, required=True, help="Directories containing six probes and campaign.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_plan, plan = read_json(args.training_plan), read_json(args.probe_plan)
    if any(training_plan[k] != plan[k] for k in ("validation_seeds", "development_seeds", "subset")):
        raise ValueError("Training and gradient cohort registrations differ")
    physical, raw, training_receipts = collect_cohort(args.training_campaigns, training_plan)
    paired, improvements, predictions, receipts = collect_fidelity_cohort(args.probe_campaigns, physical, raw, plan)
    associations = stage_associations(paired, improvements, predictions, plan)
    gradient_keys = ("iteration", "gradient_horizon", "physical_objective", "epsilon", "contact_group")
    definitions = [
        ("gradient", paired, gradient_keys, ("full_sign_agreement", "full_pearson", "state_only_sign_agreement", "state_only_pearson", "unique_states", "paired_resolved", "total_directions")),
        ("microsteps", improvements, (*gradient_keys, "reward_objective"), ("gradient_improvement", "random_improvement", "gradient_minus_random", "paired_states", "repeat_return_noise")),
        ("prediction-performance", predictions, ("iteration", "prediction_horizon"), ("scaled_mse", "physical_return", "tracking_failure_rate")),
        ("stage-associations", associations, ("horizon",), ("mse_return_pearson", "mse_sign_pearson", "mse_step_pearson")),
    ]
    statistics_by_name = {name: seed_statistics(rows, keys, fields, plan["validation_seeds"]) for name, rows, keys, fields in definitions}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows, _, _ in definitions:
        save_csv(args.output / f"{name}-per-seed.csv", rows)
        save_csv(args.output / f"{name}-cohort.csv", statistics_by_name[name])
    plot_cohort(statistics_by_name, plan, args.output)
    receipt = dict(plan=plan, plan_sha256=hashlib.sha256(args.probe_plan.read_bytes()).hexdigest(),
                   training_plan_sha256=hashlib.sha256(args.training_plan.read_bytes()).hexdigest(),
                   training_campaigns=training_receipts, probe_campaigns=receipts,
                   actual_probe_gpu_hours=sum(r["actual_gpu_hours"] for r in receipts),
                   scope="Descriptive independent-training-seed statistics; states, directions, stages and evaluation repeats are dependent, not additional seeds. FD is a numerical reference; no causal or significance claim.")
    (args.output / "fidelity-cohort.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
