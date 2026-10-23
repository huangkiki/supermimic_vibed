"""Check complete native probe recordings without treating unresolved FD as truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def validate(root, *, write_receipt=True):
    root = Path(root)
    receipt = json.loads((root / "probe.json").read_text())
    args = receipt["args"]
    bank_identity = "legacy_path_only"
    if receipt.get("bank_sha256"):
        bank_receipt = json.loads((root / "bank.json").read_text())
        if bank_receipt["bank_sha256"] != receipt["bank_sha256"]:
            raise ValueError("Bank content identity differs between receipts")
        if bank_receipt["prefix_controller"] != args["prefix_controller"] or bank_receipt["prefix_steps"] != args["prefix_steps"]:
            raise ValueError("Bank prefix differs from probe settings")
        bank_path = Path(bank_receipt["bank_path"])
        bank_identity = "sha256_recorded"
        if bank_path.is_file():
            with bank_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != receipt["bank_sha256"]:
                raise ValueError("Bank file changed since probe")
            bank_identity = "sha256_verified"
        with (root / "bank-diagnostics.csv").open() as stream:
            diagnostics = list(csv.DictReader(stream))
        if len(diagnostics) != args["samples"] or {int(r["sample"]) for r in diagnostics} != set(range(args["samples"])):
            raise ValueError("Missing or duplicate bank diagnostics")
        for row in diagnostics:
            for key in ("boundary_action_delta_l2", "following_action_delta_l2_mean"):
                if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                    raise ValueError("Invalid action-continuity recording")
    checkpoints = set(receipt["checkpoint_sha256"].values())
    samples = set(range(args["samples"]))
    epsilons = set(args["epsilons"])
    with (root / "prediction.csv").open() as stream:
        prediction = list(csv.DictReader(stream))
    with (root / "gradient.csv").open() as stream:
        gradient = list(csv.DictReader(stream))
    prediction_keys = {(r["checkpoint_sha256"], int(r["sample"]), int(r["horizon"])) for r in prediction}
    expected_prediction = {(ck, n, h) for ck in checkpoints for n in samples for h in (1, 8, 16, 32)}
    if prediction_keys != expected_prediction or len(prediction_keys) != len(prediction):
        raise ValueError(f"Incomplete or duplicate prediction records: {root}")
    gradient_keys = {(r["checkpoint_sha256"], int(r["sample"]), r["objective"], int(r["direction"]), float(r["epsilon"])) for r in gradient}
    expected_gradient = {(ck, n, objective, direction, eps) for ck in checkpoints for n in samples
                         for objective in ("hard", "soft") for direction in range(args["directions"] + 1) for eps in epsilons}
    if gradient_keys != expected_gradient or len(gradient_keys) != len(gradient):
        raise ValueError(f"Incomplete or duplicate gradient records: {root}")
    for row in prediction:
        for key in row:
            if key.endswith("mse") and (not math.isfinite(float(row[key])) or float(row[key]) < 0):
                raise ValueError(f"Invalid prediction error: {key}")
    for row in gradient:
        for key in ("predicted_derivative", "fd_derivative", "repeat_return_noise", "initial_state_max_drift",
                    "baseline_return", "positive_return", "negative_return", "improvement", "wm_gradient_l2_norm", "direction_l2_norm"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite gradient recording: {key}")
        if row["resolved"] not in ("True", "False"):
            raise ValueError("Invalid reference-resolution flag")
        if row["resolved"] == "True" and float(row["initial_state_max_drift"]) > 1e-5:
            raise ValueError("Irreproducible starting state was scored as resolved")
        if row.get("reward_objective", "full") != args.get("reward_objective", "full"):
            raise ValueError("Reward objective mismatch")
        if row.get("contact_group_level", "geometry") != args.get("contact_level", "geometry"):
            raise ValueError("Contact definition mismatch")
    first = next(iter(checkpoints))
    groups = Counter(r["contact_group"] for r in gradient if r["checkpoint_sha256"] == first
                     and int(r["direction"]) == 0 and r["objective"] == "soft" and float(r["epsilon"]) == min(epsilons))
    result = dict(status="recording_checks_passed", checkpoints=len(checkpoints), samples=len(samples),
                  bank_identity=bank_identity,
                  prediction_rows=len(prediction), gradient_rows=len(gradient), contact_samples=dict(groups),
                  max_initial_state_drift=max(float(r["initial_state_max_drift"]) for r in gradient),
                  resolved_rows=sum(r["resolved"] == "True" for r in gradient),
                  scope="Recording completeness and numerical bookkeeping; not model accuracy or independent-sample significance")
    if write_receipt:
        (root / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("probes", nargs="+")
    args = parser.parse_args()
    for path in args.probes:
        print(path, json.dumps(validate(path), allow_nan=False))


if __name__ == "__main__":
    main()
