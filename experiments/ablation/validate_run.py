"""Validate actual campaign artifacts, without equating a smoke pass with task success."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("campaign")
    args = parser.parse_args()
    root = Path(args.campaign)
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign["status"] != "completed":
        raise ValueError("campaign incomplete; preserve failure evidence")
    checks = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
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
        if cfg["method"].endswith("_frozen"):
            bootstrap = torch.load(cfg["bootstrap"], map_location="cpu", weights_only=False)
            assert all(torch.equal(value, last["wm"][key]) for key, value in bootstrap["wm"].items()), "frozen WM changed"
            assert last["ledger"]["simulator_steps"] == first["ledger"]["simulator_steps"]
            assert last["ledger"]["wm_training_seconds"] == first["ledger"]["wm_training_seconds"]
        if cfg["method"] == "ppo_sim":
            assert last["wm"] is None and last["ledger"]["imagined_steps"] == 0
        if cfg["method"] != "pretrain":
            for checkpoint in checkpoints:
                evaluation = json.loads((run / checkpoint.stem / "evaluation.json").read_text())
                per_clip = list(csv.DictReader((run / checkpoint.stem / "per_clip.csv").open()))
                assert len(per_clip) == len(manifest["motions"]) == evaluation["total_clips"]
                assert {r["motion"] + ".csv" for r in per_clip} == set(manifest["motions"])
                assert all(math.isfinite(float(r["normalized_return"])) for r in per_clip)
                assert all(r["failed"] == "True" or int(r["steps"]) == int(r["reference_steps"]) for r in per_clip)
        checks.append(dict(method=cfg["method"], seed=cfg["seed"], iterations=cfg["iterations"],
                           actor_weights_changed=changed, training_and_full_physical_evaluation_valid=True))
    result = dict(checks=checks, actual_gpu_hours=campaign["actual_gpu_hours"],
                  scope="execution and recording qualification, not policy convergence or scientific superiority")
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
