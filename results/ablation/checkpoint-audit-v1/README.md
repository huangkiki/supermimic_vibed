# Fixed-policy WM checkpoint and gradient comparison

[中文实验报告](../../../docs/ablation/checkpoint-audit-results.md) · [Protocol](../../../docs/ablation/checkpoint-audit-protocol.md)

Five WM snapshots; one fixed Gaussian policy and critic; H32; two run starts; one training seed. The independent physical score reference supports conditional full-vector squared-error comparisons. It is not an analytic simulator Jacobian or a full PPO training benchmark.

- `raw-data.tar.gz`: native CSV/JSON, rollout/calibration tapes, source and resource receipts, both failures and CPU vector-validation receipts.
- `artifact-manifest.json`: archive/file hashes and the paths/hashes of large mean-gradient tensors retained on the native host.
- `analysis/`: prediction, finite-difference uncertainty and equal-parameter-step comparisons.
- `calibrated-analysis/`: full-gradient error differences, approximate simultaneous intervals, equal-calibration-KL physical updates and unresolved cells.
- `resource-accounting.csv`: all five allocations, including failures; total0.564807806 GPU-hours.

The public archive reproduces all published CSVs and figures on CPU. Rechecking the large-vector receipt independently requires the retained tensors. No motion assets or training checkpoints are redistributed here. See the report for exact reproduction commands and scope.
