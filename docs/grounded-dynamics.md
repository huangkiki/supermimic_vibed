# Grounding the dynamics representation

This experiment transfers the **state alignment and inverse dynamics objectives**
from [GAWM](https://github.com/zsibot/gawm/blob/c8e50746e4f21c660d6dfaec97d5164db541b1e6/train.py)
to SuperMimic's state-based dynamics model. It is not a port of GAWM's visual JEPA
or CEM planner. SuperMimic already supervises physical state predictions; adding
a second copy of that loss without a shared learned representation would not
transfer GAWM's mechanism.

An optional encoder maps the existing heading-invariant state features to a
latent vector. With `--wm_state_skip`, the dynamics MLP retains the original
normalized physical inputs and appends the learned vector. Without that flag,
the vector replaces the physical state features, preserving the first experiment's
checkpoint layout. The dynamics MLP also receives the executed action and
physical context, and predicts the original frame-local state increments. Two
training-only heads regularize the encoder:

- State alignment: decode the normalized successor state features from the pair
  of current and successor encodings.
- Inverse dynamics: decode the normalized **executed action** from that same pair
  and physical context. Context accounts for randomized actuator offsets. The
  encoder and inverse head never receive the target action as an input.

Both objectives backpropagate into the state encoder used by the dynamics
model. They use real replay transitions, including the terminal successor before
an automatic reset. Transitions after the first termination in a sampled window
are masked using the existing replay mask. Feature and action normalization uses
the existing input statistics, without a second competing normalizer.

The physical rollout loss, integration, actor, critic, reward, termination,
control rate and evaluation protocol are unchanged. Policy optimization still
freezes the dynamics parameters while differentiating through its state and
action inputs. Auxiliary heads are absent from policy deployment.

## Pilot evidence and current hypothesis

A seed-0 pilot at commit `ebff203` trained each condition for 1,000 iterations
using 512 environments, world-model batch 256 and policy batch 512. All used
1,312,768 simulator transitions and the unchanged full-40-clip evaluation:

| Model | Normalized return | Failed clips |
|---|---:|---:|
| Original | 0.4309 | 15/40 |
| Encoder replacing physical inputs | 0.3963 | 19/40 |
| Same encoder with SA=0.1 and IDM=0.1 | 0.3954 | 20/40 |

Neither encoder variant improved tracking in this pilot. On identical native
reference-PD probes, normalized 8-step prediction error was 8.84, 26.29 and
25.07 respectively. The common probe uses the same target scales and transitions
for every model. It samples the training motion library, so it does not measure
unseen-motion generalization. [Full scores and configuration](../results/gawm-pilot/summary.json).

The next hypothesis is that retaining the known physical state can avoid the
learning cost of replacing it with an encoder. `--wm_state_skip` implements this
direct input path; its tracking benefit is **not yet established**. Its CPU/CUDA
checks and native train/resume check pass. The default remains the original model.

## Compare retained-state features

Use the setup and data preparation in the repository README. The original model
remains the default and accepts the original checkpoint parameter layout.
The following are **experimental configurations**, not established improvements:

```bash
# Original physical-state MLP
python supermimic/train.py --log_dir logs/original --seed 0

# Physical inputs plus learned features, without auxiliary supervision
python supermimic/train.py --log_dir logs/latent --seed 0 \
  --wm_latent_dim 128 --wm_state_skip

# Same representation with both GAWM-inspired objectives
python supermimic/train.py --log_dir logs/grounded --seed 0 \
  --wm_latent_dim 128 --wm_state_skip --wm_state_weight 0.1 --wm_idm_weight 0.1
```

Apply identical overrides for environment count, batch sizes, replay capacity,
number of updates and collection budget to all three conditions. The latent and
grounded conditions have identical architecture and initialization at a given
seed. The original condition is a system comparison: its architecture and random
initialization consumption differ. For loss attribution, compare the latter two.

`train_log.csv` retains its original columns and physical rollout loss semantics.
`grounding_log.csv` records the state and inverse-dynamics losses separately. Each
loss averages over feature dimensions, sums valid steps and divides by batch size,
just like the physical rollout objective. Reported values divide by window length.
Checkpoint arguments record architecture and loss weights; peak allocated and reserved
PyTorch GPU bytes are also saved (these do not include simulator allocations). When resuming, pass
matching arguments, including `--wm_latent_dim`, `--wm_state_skip` and the grounding weights.
Replay is not checkpointed, so a resumed run collects a fresh warm-up buffer.

Evaluate with the existing `supermimic/evaluate.py`: all 40 clips, first-frame
starts, no observation/action noise or domain randomization, original failure
thresholds. A lower auxiliary loss alone is not evidence of improved tracking.
Report per-clip failures and normalized return alongside measured training time,
peak GPU memory and sample counts. Short runs do not establish the README's
original 0.82 return / 40-of-40 target or real-robot performance.

## Checks without Isaac Lab

State tensor operations and replay are independent of simulator imports. With
PyTorch installed, run:

```bash
python -m unittest discover -s tests -v
SM_TEST_DEVICE=cuda SM_TEST_COMPILE=1 python -m unittest discover -s tests -v
```

GitHub Actions runs the CPU tensor checks. The CUDA compile check requires a C compiler and the matching CUDA-enabled
PyTorch environment. Checks cover checkpoint compatibility, gradient paths,
action-input isolation, heading invariance and terminal/reset handling.
