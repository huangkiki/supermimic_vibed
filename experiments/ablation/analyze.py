"""Build comparison CSV/plots only from qualified physical evaluation records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def comparison_key(record):
    manifest = record["manifest"]
    cfg = manifest["config"]
    return json.dumps(dict(gpu=record["evaluation_gpu"], train_gpu=manifest["gpu"],
                           torch=manifest["torch"], cuda=manifest["cuda"],
                           packages=manifest.get("packages", {}),
                           motions=manifest["motions"], assets=manifest["assets"],
                           protocol=record["protocol"],
                           core_sources={k: v for k, v in manifest["source_hashes"].items() if k.startswith("supermimic/")},
                           config={k: cfg[k] for k in ("subset", "horizon", "hist", "goal_frames", "gamma", "lam", "amp", "num_envs", "pi_batch")}), sort_keys=True)


def collect_records(paths):
    records = [json.loads(Path(p).read_text()) for p in paths]
    if not records:
        raise ValueError("No physical evaluations found; refusing to fabricate curves")
    if len({comparison_key(r) for r in records}) != 1:
        raise ValueError("Mixed GPU/data/physics/precision/evaluation protocols: analyze separately")
    if any(r["evaluation_gpu"] != r["manifest"]["gpu"] for r in records):
        raise ValueError("Training/evaluation GPUs differ")
    groups = defaultdict(list)
    seen = set()
    for record in records:
        key = (record["method"], record["seed"], record["iteration"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation {key}; keep repeatability audits separate")
        seen.add(key)
        groups[key[:2]].append(record)
    rows = []
    for (method, seed), group in groups.items():
        eval_seconds, eval_steps = 0., 0
        for record in sorted(group, key=lambda r: r["iteration"]):
            eval_seconds += record["evaluation_seconds"]
            eval_steps += record.get("evaluation_simulator_steps", 0)
            ledger = record["train_ledger"]
            rows.append(dict(method=method, seed=seed, iteration=record["iteration"],
                             physical_return=record["physical_return"], tracking_failure_rate=record["tracking_failure_rate"],
                             train_wall_seconds=ledger["train_wall_seconds"],
                             end_to_end_seconds=ledger["train_wall_seconds"] + eval_seconds,
                             total_gpu_hours=ledger["train_gpu_hours"] + eval_seconds / 3600,
                             simulator_steps=ledger["simulator_steps"], imagined_steps=ledger["imagined_steps"],
                             evaluation_simulator_steps=eval_steps, total_simulator_steps=ledger["simulator_steps"] + eval_steps,
                             wm_training_seconds=ledger["wm_training_seconds"],
                             policy_optimization_seconds=ledger["policy_optimization_seconds"],
                             collection_seconds=ledger["collection_seconds"], cumulative_evaluation_seconds=eval_seconds,
                             checkpoint_sha256=record["checkpoint_sha256"]))
    return rows


def aggregate_curves(rows, x_key):
    methods = defaultdict(lambda: defaultdict(list))
    for row in rows:
        methods[row["method"]][row["seed"]].append(row)
    aggregate = []
    for method, seeds in methods.items():
        series = []
        for seed, points in seeds.items():
            points = sorted(points, key=lambda r: (r[x_key], r["iteration"]))
            series.append((seed, points))
        lower = max(points[0][x_key] for _, points in series)
        upper = min(points[-1][x_key] for _, points in series)
        if upper < lower:
            continue
        grid = sorted({r[x_key] for _, points in series for r in points if lower <= r[x_key] <= upper})
        for x in grid:
            # Last observed checkpoint (no interpolation across missing training outcomes).
            # For frozen WM Simulator Steps are constant: emit all raw points in the plot;
            # no learning-vs-simulator-samples curve can be inferred from that vertical line.
            values = [max((r for r in points if r[x_key] <= x), key=lambda r: (r[x_key], r["iteration"]))["physical_return"]
                      for _, points in series]
            aggregate.append(dict(method=method, x_axis=x_key, x=x, mean=float(np.mean(values)),
                                  std=float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                                  seeds=len(values)))
    return aggregate


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Physical simulator evaluation — pilot, not convergence evidence")
    args = parser.parse_args()
    rows = collect_records(args.evaluations)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "evaluations.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    aggregate = []
    palette = dict(ppo_sim="#4477AA", ppo_wm_online="#EE6677", supermimic="#228833",
                   ppo_wm_frozen="#AA3377", bptt_wm_frozen="#CCBB44")
    for ax, x_key, label, scale in zip(axes,
            ("end_to_end_seconds", "total_gpu_hours", "simulator_steps"),
            ("End-to-end wall-clock (minutes)", "Total allocated GPU-hours", "Training simulator steps"),
            (60., 1., 1.)):
        stats = aggregate_curves(rows, x_key)
        aggregate.extend(stats)
        for method in sorted({r["method"] for r in rows}):
            group = sorted((r for r in stats if r["method"] == method), key=lambda r: r["x"])
            color = palette.get(method)
            for seed in sorted({r["seed"] for r in rows if r["method"] == method}):
                raw = sorted((r for r in rows if r["method"] == method and r["seed"] == seed), key=lambda r: r["iteration"])
                ax.plot([r[x_key] / scale for r in raw], [r["physical_return"] for r in raw],
                        color=color, alpha=.35, marker=".", linewidth=.8)
            if group:
                x = np.array([r["x"] / scale for r in group])
                y = np.array([r["mean"] for r in group])
                ax.step(x, y, where="post", color=color, label=f"{method} (n={group[0]['seeds']})")
                if group[0]["seeds"] >= 2:
                    sd = np.array([r["std"] for r in group])
                    ax.fill_between(x, y - sd, y + sd, step="post", color=color, alpha=.15)
        ax.set(xlabel=label, ylabel="Full-clip normalized physical return")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.savefig(output / "return-compute.png", dpi=180)
    fig.savefig(output / "return-compute.svg")
    save_csv(output / "curve-statistics.csv", aggregate)
    (output / "analysis.json").write_text(json.dumps(dict(evaluation_count=len(rows),
        seeds=sorted({r["seed"] for r in rows}), aggregation="sample SD, left hold at common support, no extrapolation",
        accounting="full bootstrap charged per method; all preceding checkpoint evaluations charged; no cross-GPU mixing",
        caution="equal iterations are not equal compute; frozen WM has constant simulator steps"), indent=2) + "\n")


if __name__ == "__main__":
    main()
