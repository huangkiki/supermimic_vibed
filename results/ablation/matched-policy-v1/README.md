# Matched BPTT/PPO parameter-gradient estimates

[Protocol](../../../docs/ablation/matched-policy-estimators.md) · [中文结果](../../../docs/ablation/matched-policy-results.md)

The corrected `h1/h8/h32-full-02` cases share one checkpoint, two run starts, Gaussian σ=.1, 16 gradient-noise tapes and 16 disjoint physical-reference tapes. These are **noise samples, not training seeds**. The comparison is the first on-policy PPO surrogate gradient (lambda=1, unnormalized advantages) against reparameterized BPTT; physical steps have equal parameter L2 displacement. It is not a comparison of complete multi-epoch optimizers.

`raw-data.tar.gz` preserves small artifacts from initial qualification, the failed H32 probe, its independent numerical audit, and corrected cases. It includes raw physical-reference tensors, source commands, logs, per-probe bank/CSV/JSON, external process receipts and the complete tensor SHA256/size manifest. Failures remain in the cumulative cost. `analysis/` contains separately validated corrected measurements and the plot.

CPU-only verification available from the repository package:

```bash
python -m unittest discover -s tests -p test_native_estimator_publication.py -v
```

This verifies published file hashes, shared starts/history, physical finite differences, sampling SE, the numerical counterexample and all process costs. Full gradient vectors are retained on `a1008` under `/home/huangyue/work/supermimic-ablation-20261023/runtime/matched-policy-v1`, with hashes in `all3-tensor-validation.json` inside the archive. The small publication package **does not include** `estimate-*.pt` or `mean-gradients.pt`; it cannot independently recalculate gradient variance without those files. Do not mistake the retained tensor-validation receipt for a new local tensor check.

To reproduce all raw-gradient checks and figures, restore the complete corrected case directories and verify their size/SHA256 before loading trusted `.pt` files:

```bash
python -m experiments.ablation.analyze_policy_estimators \
  "$RAW"/h{1,8,32}-full-02 --output "$NEW_ANALYSIS_DIRECTORY"
```

Use Bash brace expansion and a new output directory. The module verifies raw gradient means/variance, held-out physical statistics, direction predictions, bank identity, precision and GPU before plotting. The figure uses **SE across held-out noise tapes**, not numerical-repeat SD or training-seed intervals. Internal per-estimator timers exclude artifact writes and interpreter setup; complete process allocation costs are preserved separately in `cost-reconciliation.json` and its source receipts.

Native execution source is `3c294d1cdf6e1153da2e79b9292e2b586b69c177`; initial qualification and the failure used `7a4df20`. The deterministic parent diagnostics and formal training queue use their own immutable source snapshots. The historical 0.5 GPU-hour cumulative diagnostic cap includes the prior deterministic batch, initial attempts, failure and audit; it is not permission to start another experiment without a fresh resource check.
