# Native policy-parameter gradient diagnostics

[中文结果、限制与资源账本](../../../docs/ablation/policy-gradient-native-v1.md)

15 completed probes, one checkpoint (training seed0), two fixed run states. Twelve training-precision cases (H1/8/32 × critic × direct-action backward path), three full-objective FP32 cases. No independent-training-seed interval and no PPO gradient comparison yet.

- `raw-data.tar.gz`: 86 files, including per-probe bank/CSV/summary/receipt/stdout, four external process/campaign receipts, executed controllers and the analysis-source snapshot.
- `tensor-validation.json`: CPU verification of all 15 complete probes, with SHA256 and bytes for the six artifacts per probe. This is retained execution evidence, not a substitute for loading the tensors again.
- `training/`, `fp32/`: separately generated CSV, PNG/SVG and analysis receipts. `probe_main_seconds` excludes interpreter/startup overhead; use the external allocation receipts for the **0.169489818 GPU-hour** total. Inherited costs in campaign receipts must not be summed again.
- `per-start-updates.csv` describes individual physical outcomes, not per-state predictions of the minibatch gradient. `directions.csv` describes the shared batch-mean target.

The CI test below re-extracts the small archive, verifies its file hashes and identical raw histories, recomputes finite differences/resolution/summary metadata, keeps precision groups separate, and reconciles all 15 process costs. It **does not run physics or revalidate missing large tensors**:

```bash
python -m unittest discover -s tests -p test_native_policy_publication.py -v
```

For full tensor validation and figure regeneration, use Python with this repository, Torch, NumPy and Matplotlib. On the experiment host the root is `/home/huangyue/work/supermimic-ablation-20261023/runtime/policy-gradient-v1`; the verified local mirror is stored under `/media/robdog/data/CodexArchives/supermimic-ablation-20261023/policy-gradient-native-v1` (copy status must be checked before use). The raw originals remain on the host. Each case requires `gradients.pt`, `direction-rollouts.pt`, `policy-bank.pt`, `directions.csv`, `summary.json`, `probe.json`. Recover the small files from the archive and the two large tensors per case from the retained store, checking the manifest SHA256/size before loading trusted `.pt` files.

```bash
# Set RAW to a COMPLETE extracted dataset, not just the small archive.
python -m experiments.ablation.analyze_policy_fidelity \
  "$RAW"/h{1,8,32}-{full,no-critic,detach-action,no-critic-detach-action}-01 \
  --output "$NEW_OUTPUT/training"
python -m experiments.ablation.analyze_policy_fidelity \
  "$RAW"/h{1,8,32}-full-fp32-01 --output "$NEW_OUTPUT/fp32"
```

Bash brace expansion is used above. Output directories must be new. The analyzer checks complete pairs, numerical resolution, interpretation labels, raw vectors, saved physical returns and restore drift, and exact shared start/history tensors. Regenerated PNG/SVG metadata may differ; CSV quantities are the reproducible measurements.

To rerun physics, use the pinned source `19305b7`, the original checkpoint and bank hashes in the report, and the qualified [native environment](../../../docs/ablation/environment.md). The initial command is in [the protocol](../../../docs/ablation/policy-gradient-fidelity.md); the archive preserves all 15 executed commands, resource preflights, CPU affinities, hard outer timeouts and the cumulative cap. The first run reconstructs history; every later run uses its `policy-bank.pt`. Recheck resources before running; the historical 0.25 GPU-hour cap is not a current allocation.
