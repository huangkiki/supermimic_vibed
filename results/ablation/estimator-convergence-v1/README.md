# Independent-direction BPTT / PPO convergence

[中文结果报告](../../../docs/ablation/estimator-convergence-results.md) ·
[Frozen protocol](../../../docs/ablation/estimator-convergence.md) ·
[Machine-readable plan](../../../experiments/ablation/configs/estimator-convergence-v1.json)

The experiment holds one seed 0 checkpoint, two saved run starts, the WM, critic,
normalizers, Gaussian standard deviation 0.1 and H32 fixed. Eight unit-L2
Rademacher parameter directions are generated with seed 50001 before observing
any gradient. Nested estimator noise samples 16/64/256 use seeds 30001–30256;
32 disjoint reference tapes use seeds 40001–40032. All three parameter scales
0.003, 0.01 (primary), 0.03 are retained. Noise tapes are not training seeds.

PPO is the first score-function gradient at the old policy, with lambda=1,
unnormalized advantages and a fixed critic baseline. Both estimators target the
same WM soft-survival objective with critic bootstrap. Physical soft+critic
finite differences provide a matched numerical reference. Hard physical return
is a different objective and is not used to claim derivative accuracy.

## Files and independent checks

`raw-data.tar.gz` contains both native phases, all logs, complete command and
resource receipts, compact per-draw gradient projections, raw per-start returns,
parameter displacements, packed direction signs, gradient norms and streaming
variance statistics. It includes the original input bank and frozen plan under
`inputs/`. `artifact-manifest.json` binds every published file and the archive to
SHA256 and byte counts, and identifies the larger retained mean vectors.

`analysis/` contains CPU-recomputed CSVs, three figures in PNG/SVG, paired
estimator/model gap statistics and a validation receipt. `resource-accounting.csv`
counts all five native allocations, including the three post-hoc repeats below. Full-vector means were checked against every
published direction projection and their reported norms using the actual large
files; the receipt records that check. Per-draw full vectors were not retained.

Recompute the compact publication on CPU, using fresh extraction/output paths:

```bash
mkdir /absolute/path/to/convergence-raw
tar -xzf results/ablation/estimator-convergence-v1/raw-data.tar.gz \
  -C /absolute/path/to/convergence-raw
python -m experiments.ablation.analyze_estimator_convergence \
  /absolute/path/to/convergence-raw/experiment \
  --output /absolute/path/to/new-analysis
python -m unittest discover -s tests -p test_native_convergence_publication.py -v
```

`reference-audit.tar.gz` separately preserves three fresh-process repeats of
direction 0 with noise seeds 40027 and 40001, including its controller source,
logs and unchanged parent-budget binding. The large seed-40027 differences were
selected for this diagnostic after viewing primary data. These are not additional
independent Monte Carlo samples and never replace the frozen primary results.
Extract that archive into a separate directory and add
`--reference-audit /absolute/path/to/audit-raw` to the analysis command to reproduce
`reference-audit.csv`. The publication tests verify both archives and all costs.

Without the large mean vectors, the compact package checks raw return/projection
statistics, direction identities, variance conservation, hashes and allocation
costs; it cannot independently reconstruct the full gradient mean. To repeat
the additional vector check, obtain the immutable `experiment/final-means.pt`
from the experiment host, verify its manifest hash, and add `--means` to the
analysis command. The qualification and experiment mean vectors remain at:

```text
a1008:/home/huangyue/work/supermimic-ablation-20261023/runtime/estimator-convergence-v1/{qualification,experiment}/final-means.pt
```

The trusted checkpoint also remains on that host at
`formal-repo/logs/ablation/formal-locomotion-seed0-v1/supermimic-seed0/ckpt_004000.pt`
relative to `/home/huangyue/work/supermimic-ablation-20261023`. Its SHA256 is pinned
in the plan. It is not included in the compact publication. Native reproduction
requires that checkpoint, the original source bank, the same assets/motions and
the manifest-qualified simulator environment; see the protocol's native command.

## Accounting and limits

Native source is `5e1ac4548334a3dcbdccdaafb60683dae86905f5`. The native run uses
Torch 2.11.0+cu128, CUDA 12.8 and MuJoCo/MuJoCo-Warp 3.15.0 on one A100-PCIE-40GB.
The exact package manifest is in each `probe.json`. FP32, TF32 disabled, eager.

The one-GPU-hour cap includes qualification, initialization, serialization,
reference scans, post-hoc repeats and cleanup. The combined allocation is
0.6055532683 GPU-hours; no new budget was created for the repeats.
Per-estimator operation seconds additionally isolate
the gradient-computation path, including its imagined rollout, but are not a
production-training throughput benchmark: these are eager two-start probes.
No policy update or WM training occurs in this campaign. Existing checkpoint
training costs belong to the earlier training campaign and are not claimed to
be free or included in these diagnostic times. Earlier diagnostic ledgers are
unchanged.

All eight directions contribute to RMSE. Sign agreement and Pearson correlation
additionally show the reference-resolved denominator, using the preregistered
repeat/scale/sampling screens. All scales and unresolved directions remain in
the CSVs. SE is conditional on this snapshot and these starts; it is neither a
numerical-error bound nor a training-seed confidence interval. The critic was
not refit for the auxiliary Gaussian policy, and this score estimator is not
full multi-epoch PPO. These results do not establish a full-vector accuracy
ranking or a global relationship between state prediction MSE and gradient
fidelity.
