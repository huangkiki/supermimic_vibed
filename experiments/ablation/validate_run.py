"""Validate actual campaign artifacts, without equating a smoke pass with task success."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import torch

from .analyze import collect_records
from .common import sha256
from .run_suite import METHOD_ORDER


def validate_run_layout(root, campaign):
    expected = {f"{method}-seed{seed}" for method in campaign["args"].get("methods", METHOD_ORDER)
                for seed in campaign["args"]["seeds"]}
    manifests = sorted(root.glob("*/manifest.json"))
    actual = {p.parent.name for p in manifests}
    if not expected or actual != expected:
        raise ValueError(f"Campaign run coverage differs: missing={expected - actual}, extra={actual - expected}")
    return manifests


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("campaign")
    args = parser.parse_args()
    root = Path(args.campaign)
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign["status"] != "completed":
        raise ValueError("campaign incomplete; preserve failure evidence")
    checks = []
    evaluation_paths = []
    repeats = campaign["args"].get("evaluation_repeats", 1)
    for manifest_path in validate_run_layout(root, campaign):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        cfg = manifest["config"]
        complete = json.loads((run / "completion.json").read_text())
        assert complete["status"] == "completed"
        assert complete["iterations"] == cfg["iterations"]
        checkpoints = sorted(run.glob("ckpt_*.pt"))
        first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
        last = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        assert first["iteration"] == 0
        assert last["iteration"] == cfg["iterations"]
        changed = not torch.equal(first["pi"]["enc.0.weight"], last["pi"]["enc.0.weight"])
        assert changed == (cfg["method"] != "pretrain"), (run, "actor did not update as expected")
        rows = list(csv.DictReader((run / "train.csv").open()))
        assert len(rows) == cfg["iterations"]
        for row in rows:
            for name, value in row.items():
                if name not in ("method", "kl_stopped") and value:
                    assert math.isfinite(float(value)), (run, name, value)
        if "_frozen" in cfg["method"]:
            bootstrap = torch.load(cfg["bootstrap"], map_location="cpu", weights_only=False)
            assert all(torch.equal(value, last["wm"][key]) for key, value in bootstrap["wm"].items()), "frozen WM changed"
            assert last["ledger"]["simulator_steps"] == first["ledger"]["simulator_steps"]
            assert last["ledger"]["wm_training_seconds"] == first["ledger"]["wm_training_seconds"]
        if cfg["method"] == "ppo_sim":
            assert last["wm"] is None and last["ledger"]["imagined_steps"] == 0
        if cfg["method"] != "pretrain":
            for checkpoint in checkpoints:
                for repeat in range(repeats):
                    dest = run / checkpoint.stem
                    if repeats > 1:
                        dest /= f"repeat_{repeat:03d}"
                    evaluation_path = dest / "evaluation.json"
                    evaluation = json.loads(evaluation_path.read_text())
                    assert evaluation["checkpoint_sha256"] == sha256(checkpoint), "evaluation/checkpoint mismatch"
                    assert evaluation.get("evaluation_repeat_index", 0) == repeat
                    assert evaluation.get("evaluation_repeat_count", 1) == repeats
                    per_clip = list(csv.DictReader((dest / "per_clip.csv").open()))
                    assert len(per_clip) == len(manifest["motions"]) == evaluation["total_clips"]
                    assert {r["motion"] + ".csv" for r in per_clip} == set(manifest["motions"])
                    assert all(math.isfinite(float(r["normalized_return"])) for r in per_clip)
                    assert all(r["failed"] == "True" or int(r["steps"]) == int(r["reference_steps"]) for r in per_clip)
                    evaluation_paths.append(evaluation_path)
        checks.append(dict(method=cfg["method"], seed=cfg["seed"], iterations=cfg["iterations"],
                           actor_weights_changed=changed, training_and_full_physical_evaluation_valid=True))
    if evaluation_paths:
        collect_records(evaluation_paths, root / "campaign.json")
    result = dict(checks=checks, actual_gpu_hours=campaign["actual_gpu_hours"],
                  evaluation_processes=len(evaluation_paths), evaluation_repeats_per_checkpoint=repeats,
                  scope="execution and recording qualification, not policy convergence or scientific superiority")
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
