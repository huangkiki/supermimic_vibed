# H32 FP32 critic/action gradient evidence

Three additional native probes complete the H32 2×2 grid. The full-target FP32 control is reused from the prior publication, not rerun or paid twice. All inputs refer to one seed0 checkpoint, two fixed run starts and probe RNG9001. [Report](../../../docs/ablation/policy-gradient-components.md) · [Requirement-level conclusions](../../../docs/ablation/policy-gradient-conclusions.md).

- `raw-data.tar.gz`: 21 small raw files: 4 banks, run receipts, raw stdout, direction CSVs, summaries and the incremental allocation ledger. SHA256: `e9375247b665e5dc6946b65c1e232e0c7ba78c3b389ed0d8bc8a5d51e4fbdcf0`.
- `validation.json`: small-file hashes and the full gradient/rollout tensor hashes; all 18 native original-path probes passed the stronger vector validation on CPU. Full tensors remain on a1008 at the listed paths and are not in this small archive.
- `analysis/`: component norms/projections, actual Adam updates, per-start outcomes, finite differences, precision comparison, PNG/SVG figures.
- `cpu-norm-audit.jsonl`, `initial-cpu-analysis-failure.log`: retained evidence of the CPU FP32 reduction error; FP64 reference reconstruction fixes the failure without relaxing tolerances.

Native source is `19305b7`; successful tensor/component analysis is `6746e1a`. Plot width was subsequently corrected for a one-horizon figure; values are unchanged. The remaining matrix is not an independent seed cohort.

## CPU reproduction

Small-file integrity, complete four-case measurement grid and allocation accounting need no simulator or GPU:

```bash
python -m unittest discover -s tests -p test_native_component_publication.py -v
```

To recompute from the full raw tensors, set `RAW` to the retained directory (or a verified full mirror), then use fresh output paths:

```bash
python -m experiments.ablation.analyze_policy_fidelity   "$RAW/h32-full-fp32-01" "$RAW/h32-no-critic-fp32-01"   "$RAW/h32-detach-action-fp32-01" "$RAW/h32-no-critic-detach-action-fp32-01"   --output work/h32-fp32-analysis
python -m experiments.ablation.analyze_gradient_components   "$RAW/h32-full-fp32-01" "$RAW/h32-no-critic-fp32-01"   "$RAW/h32-detach-action-fp32-01" "$RAW/h32-no-critic-detach-action-fp32-01"   --output work/h32-component-analysis
```

Both commands verify bank identity, precision/GPU/sampling consistency, all raw vectors and physical measurements before publication. Gradient decomposition requires all four flags and a common horizon, layout, source, seed and target settings; Adam vectors are never decomposed additively. Return remains an H-step discounted quantity, not the full-motion normalized benchmark score.

GPU replay commands are preserved verbatim in the allocation receipt. Before a replay, check idle GPU/UUID and disk, use new output paths, and include prior costs in the same cumulative budget. Native runs cost147.468759 seconds; all parameter diagnostics including prior failures total0.351949199 GPU-hours under0.5. Historical full-target control costs are counted once. CPU analysis is separately recorded.
