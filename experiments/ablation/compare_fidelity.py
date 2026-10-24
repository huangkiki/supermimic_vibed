"""Compare full and state-only objectives on the SAME resolved random directions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def metrics(rows):
    if not rows:
        return dict(n=0, sign_agreement=None, pearson=None)
    predicted = np.array([float(r["predicted_derivative"]) for r in rows])
    reference = np.array([float(r["fd_derivative"]) for r in rows])
    correlation = (float(np.corrcoef(predicted, reference)[0, 1])
                   if len(rows) > 1 and predicted.std() > 0 and reference.std() > 0 else None)
    return dict(n=len(rows), sign_agreement=float(np.mean(np.sign(predicted) == np.sign(reference))),
                pearson=correlation)


def compare(full_path, state_path):
    roots = [Path(full_path), Path(state_path)]
    receipts = [json.loads((p / "probe.json").read_text()) for p in roots]
    for key in ("collector_sha256", "checkpoint_sha256", "gpu", "precision", "reset_protocol"):
        if receipts[0][key] != receipts[1][key]:
            raise ValueError(f"Probe identity mismatch: {key}")
    bank_hashes = [r.get("bank_sha256") for r in receipts]
    if any(digest is not None for digest in bank_hashes):
        if bank_hashes[0] != bank_hashes[1] or not isinstance(bank_hashes[0], str) or len(bank_hashes[0]) != 64:
            raise ValueError("Paired bank content hash mismatch or missing identity")
        bank_identity = "sha256"
    else:
        paths = [r["args"].get("bank") for r in receipts]
        if not paths[0] or paths[0] != paths[1]:
            raise ValueError("Unidentified banks cannot form a paired experiment")
        # Retain old exploratory receipts, explicitly weaker than content identity.
        bank_identity = "legacy_path_only"
    for key, default in (("prefix_controller", "reference_pd"), ("prefix_steps", 16)):
        if receipts[0]["args"].get(key, default) != receipts[1]["args"].get(key, default):
            raise ValueError(f"Paired bank prefix mismatch: {key}")
    if receipts[0].get("manifest", {}).get("gpu_uuid") != receipts[1].get("manifest", {}).get("gpu_uuid"):
        raise ValueError("Paired physical GPU UUID mismatch")
    for key in ("gradient_horizon", "samples", "directions", "epsilons", "contact_level", "probe_seed"):
        if receipts[0]["args"][key] != receipts[1]["args"][key]:
            raise ValueError(f"Paired probe setting mismatch: {key}")
    if [r["args"]["reward_objective"] for r in receipts] != ["full", "state_only"]:
        raise ValueError("Expected full and state_only probe folders in that order")
    tables = []
    for p in roots:
        with (p / "gradient.csv").open() as stream:
            tables.append(list(csv.DictReader(stream)))
    def key(row):
        return (row["checkpoint_sha256"], row["objective"], float(row["epsilon"]),
                int(row["sample"]), int(row["direction"]))
    keyed = [{key(r): r for r in rows if r["direction_type"] == "random_unit_l2"} for rows in tables]
    if any(len(table) != sum(r["direction_type"] == "random_unit_l2" for r in rows)
           for table, rows in zip(keyed, tables)):
        raise ValueError("Duplicate paired direction records")
    if keyed[0].keys() != keyed[1].keys():
        raise ValueError("Paired direction grid differs")
    if any(keyed[0][k]["contact_group"] != keyed[1][k]["contact_group"] for k in keyed[0]):
        raise ValueError("Contact membership differs between paired runs")
    args = receipts[0]["args"]
    iterations = {sha: int(Path(path).stem.removeprefix("ckpt_"))
                  for path, sha in receipts[0]["checkpoint_sha256"].items()}
    result = []
    for sha, objective, eps in sorted({k[:3] for k in keyed[0]}):
        all_keys = [k for k in keyed[0] if k[:3] == (sha, objective, eps)]
        for group in ("all", "stable_contact", "switch_nearby", "flight"):
            keys = [k for k in all_keys if group == "all" or keyed[0][k]["contact_group"] == group]
            if not keys:
                continue
            paired = [k for k in keys if all(d[k]["resolved"] == "True" for d in keyed)]
            row = dict(checkpoint_sha256=sha, iteration=iterations[sha], gradient_horizon=args["gradient_horizon"],
                       bank_identity=bank_identity, bank_sha256=bank_hashes[0],
                       prefix_controller=args.get("prefix_controller", "reference_pd"),
                       physical_objective=objective, epsilon=eps, contact_group=group,
                       contact_level=args["contact_level"], total_directions=len(keys),
                       unique_states=len({k[3] for k in keys}), paired_resolved=len(paired))
            for label, table in zip(("full", "state_only"), keyed):
                own = [table[k] for k in keys if table[k]["resolved"] == "True"]
                row.update({f"{label}_{name}": value for name, value in metrics([table[k] for k in paired]).items()})
                row[f"{label}_own_resolved"] = len(own)
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", help="Contains gradient-components-{full,state_only}-h{1,8,32}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for h in (1, 8, 32) for r in compare(root / f"gradient-components-full-h{h}",
                                                 root / f"gradient-components-state_only-h{h}")]
    with (output / "paired-gradient-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    # Display all declared scales at the final checkpoint; CSV retains every checkpoint/group.
    final = max(r["iteration"] for r in rows)
    for col, horizon in enumerate((1, 8, 32)):
        selected = sorted((r for r in rows if r["iteration"] == final and r["gradient_horizon"] == horizon
                           and r["physical_objective"] == "soft" and r["contact_group"] == "all"),
                          key=lambda r: r["epsilon"])
        x = [r["epsilon"] for r in selected]
        for label, color in (("full", "#2878B5"), ("state_only", "#D86027")):
            axes[0, col].plot(x, [r[f"{label}_sign_agreement"] for r in selected], marker="o", color=color, label=label)
            axes[1, col].plot(x, [r[f"{label}_pearson"] for r in selected], marker="o", color=color)
        for r in selected:
            axes[0, col].annotate(f"n={r['paired_resolved']}", (r["epsilon"], .04), fontsize=8, ha="center")
        axes[0, col].set(title=f"H={horizon}, matched soft objective", ylim=(0, 1.05), ylabel="Sign agreement")
        axes[1, col].set(ylim=(-1, 1), ylabel="Pearson correlation", xlabel="Action perturbation scale")
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f"Full and state-only reward gradients — checkpoint {final}\n"
                 "Same resolved intersection per scale; sample counts and bank provenance in CSV")
    fig.savefig(output / "paired-objectives.png", dpi=180)
    fig.savefig(output / "paired-objectives.svg")


if __name__ == "__main__":
    main()
