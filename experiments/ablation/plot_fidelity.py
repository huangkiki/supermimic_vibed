"""Plot saved probe measurements; prediction accuracy and gradient accuracy stay distinct."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("probe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluations", nargs="*", default=[])
    args = parser.parse_args()
    root, output = Path(args.probe), Path(args.output)
    prediction = list(csv.DictReader((root / "prediction.csv").open()))
    gradient = list(csv.DictReader((root / "gradient.csv").open()))
    if not prediction or not gradient:
        raise ValueError("No real probe measurements to plot")
    horizons = {int(r.get("gradient_horizon", 32)) for r in gradient}
    if len(horizons) != 1:
        raise ValueError("Gradient horizons define different objectives; plot them separately")
    gradient_horizon = next(iter(horizons))
    objectives = {r.get("reward_objective", "full") for r in gradient}
    if len(objectives) != 1:
        raise ValueError("Different reward objectives must be analyzed separately")
    reward_objective = next(iter(objectives))
    contact_levels = {r.get("contact_group_level", "geometry") for r in gradient}
    if len(contact_levels) != 1:
        raise ValueError("Different contact definitions must be analyzed separately")
    contact_level = next(iter(contact_levels))
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(root.glob("summary-*.json")):
        record = json.loads(path.read_text())
        summaries.extend(dict(checkpoint_sha256=record["checkpoint_sha256"],
                              gradient_horizon=gradient_horizon, reward_objective=reward_objective,
                              contact_group_level=contact_level, **row) for row in record["summaries"])
    if summaries:
        with (output / "gradient-metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    receipt = json.loads((root / "probe.json").read_text())
    labels = {sha: f"{Path(path).stem.replace('ckpt_', 'it ')} / {sha[:8]}"
              for path, sha in receipt["checkpoint_sha256"].items()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hashes = list(dict.fromkeys(r["checkpoint_sha256"] for r in prediction))
    colors = {h: plt.get_cmap("tab10")(i % 10) for i, h in enumerate(hashes)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for sha in hashes:
        rows = [r for r in prediction if r["checkpoint_sha256"] == sha]
        horizons = sorted({int(r["horizon"]) for r in rows})
        mean = [np.mean([float(r["scaled_mse"]) for r in rows if int(r["horizon"]) == h]) for h in horizons]
        axes[0, 0].plot(horizons, mean, marker="o", color=colors[sha], label=labels[sha])
    axes[0, 0].set(xlabel="Prediction horizon (control steps)", ylabel="Fixed-scale state MSE", title="Same held-out states/actions for every WM")
    axes[0, 0].legend(title="Checkpoint iteration / SHA", fontsize=8)
    derivative_extent = []
    for sha in hashes:
        rows = [r for r in gradient if r["checkpoint_sha256"] == sha and r["objective"] == "soft" and r["direction_type"] == "random_unit_l2" and r["resolved"] == "True"]
        axes[0, 1].scatter([float(r["fd_derivative"]) for r in rows], [float(r["predicted_derivative"]) for r in rows],
                           alpha=.35, s=12, color=colors[sha])
        derivative_extent.extend(float(r[key]) for r in rows for key in ("fd_derivative", "predicted_derivative"))
    if derivative_extent:
        limits = [min(derivative_extent + [0.]), max(derivative_extent + [0.])]
        axes[0, 1].plot(limits, limits, color="gray", linestyle="--", linewidth=.8, label="Equal derivatives")
        axes[0, 1].legend(fontsize=7)
    else:
        axes[0, 1].text(.5, .5, "No resolved numerical references", transform=axes[0, 1].transAxes, ha="center")
    axes[0, 1].axhline(0, color="gray", linewidth=.6)
    axes[0, 1].axvline(0, color="gray", linewidth=.6)
    axes[0, 1].set(xlabel="Simulator central FD (resolved scales)", ylabel="WM directional derivative", title="Matched soft objective; FD is a numerical reference")
    groups = sorted({r["contact_group"] for r in gradient})
    epsilons = sorted({float(r["epsilon"]) for r in gradient})
    # Each checkpoint/group/scale is distinct; repeated banks are not independent samples.
    for sha in hashes:
        for i, label in enumerate(groups):
            x, y = [], []
            for eps in epsilons:
                rows = [r for r in gradient if r["checkpoint_sha256"] == sha and r["contact_group"] == label and float(r["epsilon"]) == eps and r["objective"] == "hard" and r["direction_type"] == "random_unit_l2" and r["resolved"] == "True"]
                if rows:
                    x.append(eps)
                    y.append(np.mean([np.sign(float(r["fd_derivative"])) == np.sign(float(r["predicted_derivative"])) for r in rows]))
            if x:
                axes[1, 0].plot(x, y, color=colors[sha], marker=("o", "s", "^")[i % 3],
                                linestyle=("-", "--", ":")[i % 3], label=f"{sha[:8]} / {label}")
    axes[1, 0].set(xlabel="Perturbation scale", xscale="log", ylim=(0, 1.05),
                   ylabel="Sign agreement (resolved only)", title=f"Hard physical objective, {contact_level} contact groups")
    if axes[1, 0].get_legend_handles_labels()[0]:
        axes[1, 0].legend(fontsize=7)
    for sha in hashes:
        for kind, style in (("wm_gradient", "-"), ("random_unit_l2", "--")):
            means = []
            for eps in epsilons:
                rows = [r for r in gradient if r["checkpoint_sha256"] == sha and float(r["epsilon"]) == eps and r["objective"] == "hard" and r["direction_type"] == kind and float(r["initial_state_max_drift"]) <= 1e-5]
                means.append(np.mean([float(r["improvement"]) for r in rows]) if rows else np.nan)
            axes[1, 1].plot(epsilons, means, marker="o", linestyle=style, color=colors[sha],
                            label=f"{sha[:8]} / {'gradient' if kind == 'wm_gradient' else 'random'}")
    axes[1, 1].axhline(0, color="gray", linewidth=.7)
    score_label = "Physical task return improvement" if reward_objective == "full" else "State-only physical score improvement"
    axes[1, 1].set(xlabel="Action-sequence step size (unit L2 direction)", ylabel=score_label,
                   title="One small step along the WM gradient")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"{gradient_horizon}-step action-gradient diagnostic — {receipt['args']['samples']} states, "
                 f"{receipt['args']['directions']} random directions; {reward_objective} reward, no critic")
    fig.savefig(output / "prediction-gradient.png", dpi=180)
    fig.savefig(output / "prediction-gradient.svg")
    if args.evaluations:
        evaluation = {r["checkpoint_sha256"]: r for path in args.evaluations for r in [json.loads(Path(path).read_text())]}
        fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
        matched = []
        for sha in hashes:
            if sha not in evaluation:
                continue
            mse = np.mean([float(r["scaled_mse"]) for r in prediction if r["checkpoint_sha256"] == sha and int(r["horizon"]) == 32])
            score = evaluation[sha]["physical_return"]
            ax.scatter(mse, score, color=colors[sha])
            ax.annotate(sha[:8], (mse, score), fontsize=8)
            matched.append(dict(checkpoint_sha256=sha, mse32=float(mse), physical_return=score))
        if matched:
            ax.set(xlabel="32-step fixed-bank state MSE", ylabel="Full-clip physical evaluation return",
                   ylim=(min(0., min(r["physical_return"] for r in matched)),
                         max(1., max(r["physical_return"] for r in matched))),
                   title="Observational checkpoint association (not causality)")
            fig.savefig(output / "prediction-performance.png", dpi=180)
            (output / "prediction-performance.json").write_text(json.dumps(matched, indent=2) + "\n")


if __name__ == "__main__":
    main()
