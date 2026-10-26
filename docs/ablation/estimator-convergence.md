# Independent-direction estimator convergence

This preregistered follow-up asks whether the previous BPTT advantage on a
BPTT-selected direction survives directions selected independently of both
estimators, and whether more PPO samples close the gap. It is a fixed-snapshot
diagnostic, not another policy-training comparison.

## Frozen design

The machine-readable plan is
[`estimator-convergence-v1.json`](../../experiments/ablation/configs/estimator-convergence-v1.json).
Use the same trained seed0 checkpoint at update4000, the same two saved run
starts, fixed WM/critic/normalization, H32, FP32 with TF32 disabled, and Gaussian
action standard deviation0.1. Mean-network parameters are the only differentiated
parameters. No optimization or checkpoint selection occurs.

- Generate eight independent unit-L2 Rademacher parameter directions with seed
  50001 **before** computing gradients. Save their packed signs and SHA256.
- Both estimators use the same trajectories and soft-survival objective with
  fixed-critic bootstrap. Nested sample counts16,64,256 use seeds30001–30256.
  One sample is a noise tape across the fixed two-start minibatch. These are
  Monte Carlo samples, not independent training seeds or new states.
- PPO here means the **first score-function gradient at the old policy**,
  with the causal return and detached baseline; no clipping-active later epochs,
  GAE, entropy term, learned standard deviation, Adam or policy learning.
- On independent reference tapes40001–40032, evaluate central differences in
  **both WM and physical simulation**, paired at ±epsilon. Perturbation L2
  scales are0.003,0.01(primary),0.03. Physics uses native exact-state restoration
  and cold solver reset. Record actual FP32 parameter displacement and reject
  relative L2 rounding error above2%.
- The physical soft+critic objective matches the WM target algebraically.
  Hard-termination return is retained only as a different-target effectiveness
  diagnostic. Neither finite differences nor the learned critic are ground truth.

Report per-direction estimates/SE, reference/SE, sign agreement and correlation
on a shared reference-resolved subset, and RMSE over **all eight directions**.
The existing numerical screen requires return signal above five times repeat
noise(and1e-7), adjacent-scale agreement within25%, and now also reference
mean magnitude above twice its Monte Carlo SE. Report unresolved counts and all
scales; do not improve the denominator after observing results. SE describes
noise-tape variation conditional on this checkpoint and these two starts.

Record full-vector mean/variance trace with streaming FP64 moments, per-draw
gradient norms, all directional projections, raw per-start return pairs and
separate BPTT/PPO operation times. Full vectors are retained only for the final
means. Per-draw vectors are deliberately omitted to bound storage.

## Resource qualification and execution

Use one idle A10040GB, with at least16GiB free GPU memory and20.75GiB free disk.
The initial conservative peak estimate is8GiB. A disjoint qualification uses
4 gradient samples,4 reference tapes,2 directions, the same H32 and epsilon grid.
It must finish within240seconds. Only after measured phase times, multiplied by
the full design's work ratios and a1.25 safety factor, fit the remaining budget
does the full batch launch. The total **one allocated GPU-hour** cap includes
qualification, initialization, failed processes and cleanup. This is a new
user-requested campaign; the earlier diagnostic ledger remains separate and
must not be represented as reset or free. No training/data-learning cost is
claimed here: existing checkpoint costs belong to the earlier training campaign.

Run from a clean committed checkout in the already-qualified native environment:

```bash
python -m experiments.ablation.run_estimator_convergence \
  --plan experiments/ablation/configs/estimator-convergence-v1.json \
  --checkpoint /absolute/path/to/ckpt_004000.pt \
  --bank /absolute/path/to/policy-bank.pt \
  --gpu 2 --output /absolute/path/to/new-convergence-output
```

The plan pins the input SHA256 and GPU UUID. A different machine/input requires
an explicitly revised plan, not bypassing verification. The controller records
commands, allocation, source commit and qualification. Native receipts verify
training assets, packages, core source and input hashes before/after execution.

CPU regression checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q experiments/ablation
```

## Interpretation decided before execution

If increased PPO sampling approaches the WM reference while WM and physical
references differ, this points to estimator noise plus model mismatch, not a
uniquely correct BPTT gradient. If both approach the same physical reference but
BPTT needs less work, that supports a local estimation-efficiency advantage.
If finite differences or PPO remain unresolved, report that limitation; reaching
256 samples is not itself proof of convergence. These eight projections cannot
certify the accuracy of an entire nine-million-parameter gradient. This probe
does not establish a global prediction-MSE/gradient-fidelity relationship or
explain every difference in full SuperMimic-vs-PPO training curves.
