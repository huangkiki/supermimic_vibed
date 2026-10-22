"""Plot measured tracking results for each seed; no interpolation or uncertainty claims."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = {
    "original": ("Original", "#44515f"),
    "skip-latent": ("Physical state + learned features", "#147d92"),
    "skip-grounded": ("Same features + SA / IDM", "#b55d28"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=4000)
    args = parser.parse_args()
    runs = sorted((json.loads(path.read_text()) for path in args.results), key=lambda run: run["seed"])
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Each seed must appear once")
    budgets = set()
    for run in runs:
        for condition in CONDITIONS:
            checkpoint = run["conditions"][condition]["checkpoints"][str(args.iteration)]
            evaluation = checkpoint["evaluation"]
            if not math.isfinite(evaluation["norm_return"]):
                raise ValueError("Nonfinite return")
            if not 0 <= evaluation["n_failed"] <= 40 or len(evaluation["per_clip"]) != 40:
                raise ValueError("Expected evaluation of all 40 motion clips")
            budgets.add(checkpoint["cost"]["sim_steps"])
    if len(budgets) != 1:
        raise ValueError("Compared checkpoints have different sampling budgets")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.axisbelow": True})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": (1.2, 1)})
    positions = np.arange(len(runs))
    width = 0.23
    max_failures = 0
    for index, (condition, (label, color)) in enumerate(CONDITIONS.items()):
        evaluations = [run["conditions"][condition]["checkpoints"][str(args.iteration)]["evaluation"] for run in runs]
        x = positions + (index - 1) * width
        returns = [evaluation["norm_return"] for evaluation in evaluations]
        failures = [evaluation["n_failed"] for evaluation in evaluations]
        max_failures = max(max_failures, *failures)
        bars = axes[0].bar(x, returns, width * 0.93, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{value:.4f}" for value in returns], padding=5, fontsize=10)
        bars = axes[1].bar(x, failures, width * 0.93, color=color)
        axes[1].bar_label(bars, labels=[str(value) for value in failures], padding=5, fontsize=10)
    axes[0].set(title="Tracking return", ylabel="Mean normalized return (higher is better)", ylim=(0, 1))
    axes[1].set(title="Failed motion clips", ylabel="Count out of 40 (lower is better)", ylim=(0, max(6, max_failures + 2)))
    axes[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for axis in axes:
        axis.set_xticks(positions, [f"Seed {run['seed']}" for run in runs])
        axis.grid(axis="y", color="#e4e8eb", linewidth=0.8)
        axis.margins(x=0.18)
    figure.suptitle(f"Retained-state dynamics: {args.iteration:,} training iterations", x=0.07, ha="left", fontsize=17, fontweight="bold")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.075), ncols=3, frameon=False, fontsize=10)
    figure.text(0.07, 0.035, f"Matched budget: {next(iter(budgets)):,} simulator transitions per run. All 40 training-library clips; not unseen-motion generalization.", fontsize=9, color="#4b5563")
    figure.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.24, wspace=0.29)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
