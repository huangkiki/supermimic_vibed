# Retained-state formal experiment

This protocol tests whether retaining normalized physical dynamics inputs and
appending a learned 128-dimensional state representation improves SuperMimic at
matched simulation and optimization budgets.

The primary arms are:

| Arm | Flags |
| --- | --- |
| Original | default arguments |
| Retained | `--wm_latent_dim 128 --wm_state_skip` |

The state-alignment and inverse-dynamics weights remain zero. Prior two-seed
evidence found those auxiliary losses below retained state, so they are not a
primary condition in this confirmatory run.

## Screen phase

Run three new training seeds (`2`, `3`, `4`) for both arms for 4,000 iterations:

```bash
python -m experiments.retained_state.run_formal \
  --phase screen \
  --output logs/retained-state-formal/screen \
  --python /path/to/isaaclab/python \
  --gpu 0 \
  --motion-dir data/lafan1_g1 \
  --max-gpu-hours 6
```

Every run uses 4,096 environments, world-model batch 2,048, policy batch 4,096,
`wm_window=8`, `horizon=32`, and 500 world-model-only iterations. Each final
checkpoint is evaluated on all 40 LAFAN clips with the original first-frame,
near-nominal-physics protocol.

The screen gate passes only when:

- all six training/evaluation runs complete;
- retained mean normalized return is at least 3% above original;
- retained failures do not exceed original failures;
- at least two of three seeds have non-decreasing return and no more failures.

The machine-readable decision is `results.json:gate`.

If the process is interrupted after training or evaluation artifacts were written,
rerun the exact same command with `--resume`. Completed checkpoints are not
retrained. Completed simulator stdout is parsed and converted into the missing
machine-readable metric file; the original source commit and accrued GPU-hour
budget remain in `manifest.json`.

## Final phase

Run this only after the screen gate passes:

```bash
python -m experiments.retained_state.run_formal \
  --phase final \
  --output logs/retained-state-formal/final \
  --python /path/to/isaaclab/python \
  --gpu 0 \
  --motion-dir data/lafan1_g1 \
  --max-gpu-hours 12
```

The final phase uses 14,000 iterations and the same arms, seeds, environment,
batches, checkpoint interval, metrics, and gate. Do not retune after seeing
screen results.

## Evidence and boundaries

Each output directory contains:

- `manifest.json`: source/data identity, GPU identity, cap, timings and status;
- `partial-results.json`: crash-safe completed results;
- `results.json`: complete per-seed metrics and gate;
- `runs/<phase>-<arm>-seedN/stdout.log` and `evaluation.log`;
- checkpoint SHA-256 and every failed clip.

These experiments use the training motion library. They are not unseen-motion
generalization, real-robot, or Sim2Real evidence. Three seeds are a bounded
engineering replication and do not justify a population-level statistical
claim.
