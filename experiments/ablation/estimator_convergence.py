"""Independent-direction gradient convergence at one immutable policy snapshot.

Only projections are retained per draw. Full-vector means and a scalar Welford
variance trace use O(parameters) memory, rather than O(draws * parameters).
Both WM and physics references use held-out, paired noise. Hard return is saved
as an effectiveness diagnostic, never as the derivative of the soft objective.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import save_csv
from .common import sha256, write_json
from .fidelity_math import agreement, reference_quality
from .policy_estimators import estimate_pair
from .policy_fidelity import displaced_parameters, parameter_layout, parameter_vector


METHODS = ('bptt', 'ppo')


def validate_design(gradient_seeds, reference_seeds, counts, epsilons, directions, direction_seed):
    for seeds, maximum in ((gradient_seeds, 1024), (reference_seeds, 64)):
        if not 2 <= len(seeds) <= maximum or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
            raise ValueError('Require distinct nonnegative noise seeds within the bounded design')
    if set(gradient_seeds) & set(reference_seeds) or direction_seed in set(gradient_seeds) | set(reference_seeds):
        raise ValueError('Direction, estimator and reference seeds must be disjoint')
    if (not counts or counts != sorted(set(counts)) or counts[0] < 2
            or counts[-1] != len(gradient_seeds)):
        raise ValueError('Increasing prefix counts must end at the full gradient sample count')
    if not 1 <= directions <= 32 or direction_seed < 0:
        raise ValueError('Require 1..32 independent directions and a nonnegative seed')
    if len(epsilons) < 2 or epsilons != sorted(set(epsilons)) or any(not math.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError('Require increasing positive finite perturbations')


class VectorMoments:
    """FP64 running vector mean and covariance trace; no sample stack."""

    def __init__(self, example):
        self.count = 0
        self.mean = torch.zeros_like(example, dtype=torch.float64)
        self.m2_trace = self.mean.new_zeros(())

    def add(self, vector):
        vector = vector.double()
        if vector.shape != self.mean.shape or not torch.isfinite(vector).all():
            raise ValueError('Invalid finite gradient vector')
        self.count += 1
        delta = vector - self.mean
        self.mean.add_(delta / self.count)
        self.m2_trace.add_(delta @ (vector - self.mean))

    def summary(self):
        if self.count < 2:
            raise ValueError('Variance needs at least two samples')
        trace = float(self.m2_trace / (self.count - 1))
        return dict(samples=self.count, mean_l2=float(self.mean.norm()), variance_trace=trace,
                    mean_standard_error_l2=math.sqrt(max(0., trace) / self.count))


def random_directions(parameters, count, seed):
    """Predeclared isotropic Rademacher directions, independent of all gradients."""
    signs = np.random.default_rng(seed).integers(0, 2, size=(count, parameters), dtype=np.uint8)
    vectors = (signs.astype(np.float64) * 2 - 1) / math.sqrt(parameters)
    return signs, vectors


def _mean_se(values, axis):
    return values.mean(axis=axis), values.std(axis=axis, ddof=1) / math.sqrt(values.shape[axis])


def analyze(raw, output):
    """Recompute every directional statistic from compact raw returns/projections."""
    output = Path(output)
    projections = raw['projections']  # method, draw, independent direction
    epsilons = raw['epsilons']
    counts = raw['counts'].astype(int)
    rows, summaries = [], []
    for source in ('wm', 'physical_proxy', 'physical_hard'):
        # direction, epsilon, sign(+/-), reference noise, start
        returns = raw[source].mean(axis=-1)
        differences = returns[:, :, 0] - returns[:, :, 1]
        derivatives = differences / (2 * epsilons[None, :, None])
        reference, reference_se = _mean_se(derivatives, axis=-1)
        noise = float(np.max(np.abs(raw[source + '_repeats'].mean(axis=-1)
                                    - raw[source + '_repeats'][0].mean())))
        stable = np.array([reference_quality(ref, delta.mean(axis=-1), noise)
                           for ref, delta in zip(reference, differences)])
        resolved = stable & (np.abs(reference) > 2 * reference_se)
        for count in counts:
            for method_index, method in enumerate(METHODS):
                predicted, predicted_se = _mean_se(projections[method_index, :count], axis=0)
                for e, epsilon in enumerate(epsilons):
                    errors = predicted - reference[:, e]
                    mask = resolved[:, e]
                    # Only WM and matching soft physics are accuracy comparisons.
                    interpretation = ('different hard target; directional effectiveness only' if source == 'physical_hard'
                                      else 'projected derivative error against a finite-sample numerical reference')
                    summary = dict(source=source, method=method, samples=int(count), epsilon=float(epsilon),
                        **agreement(reference[:, e], predicted, mask),
                        all_direction_rmse=float(np.sqrt(np.mean(errors ** 2))),
                        resolved_rmse=float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else None,
                        estimator_projection_se_rms=float(np.sqrt(np.mean(predicted_se ** 2))),
                        reference_se_rms=float(np.sqrt(np.mean(reference_se[:, e] ** 2))),
                        interpretation=interpretation)
                    summaries.append(summary)
                    for d in range(len(predicted)):
                        rows.append(dict(source=source, method=method, samples=int(count), direction=d,
                            epsilon=float(epsilon), prediction=float(predicted[d]), prediction_se=float(predicted_se[d]),
                            reference=float(reference[d, e]), reference_se=float(reference_se[d, e]),
                            numerical_noise=noise, epsilon_stable=bool(stable[d, e]), resolved=bool(mask[d]),
                            difference=float(errors[d]), interpretation=interpretation))
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / 'directional-comparison.csv', rows)
    save_csv(output / 'convergence.csv', summaries)
    return summaries


def run_convergence(probe, physical, output, *, gradient_seeds, reference_seeds, counts,
                    epsilons, directions, direction_seed, repeats=3, check_budget=lambda: None):
    validate_design(gradient_seeds, reference_seeds, counts, epsilons, directions, direction_seed)
    if repeats < 2:
        raise ValueError('Require numerical baseline repeats')
    output = Path(output)
    original_seed = probe.seed
    before = parameter_vector(probe.policy)
    signs, vectors = random_directions(len(before), directions, direction_seed)
    # Save direction identity BEFORE observing any estimate or reference return.
    np.savez(output / 'directions.npz', packed_signs=np.packbits(signs, axis=-1),
             parameter_count=len(before), seed=direction_seed)
    direction_vectors = torch.as_tensor(vectors, device=probe.device, dtype=torch.float64)
    design = dict(gradient_seeds=gradient_seeds, reference_seeds=reference_seeds, counts=counts,
        epsilons=epsilons, directions=directions, direction_seed=direction_seed,
        directions_sha256=sha256(output / 'directions.npz'), parameter_layout=parameter_layout(probe.policy),
        horizon=probe.horizon, starts=probe.batch, action_std=probe.policy.std,
        objective='same soft survival + fixed critic bootstrap in WM and physics',
        direction_rule='independent PCG64 Rademacher signs, unit L2; generated before estimates',
        uncertainty='noise tapes conditional on this snapshot and minibatch; not training seeds',
        storage='all directional projections and raw returns; final full means; per-draw full vectors not retained')
    write_json(output / 'convergence-design.json', design)
    moments = {method: VectorMoments(before) for method in METHODS}
    projections = np.empty((2, len(gradient_seeds), directions), dtype=np.float64)
    costs, variance, norms = [], [], []
    progress = dict(status='running', gradient_draws=0, physical_rollouts=0, wm_reference_rollouts=0)
    def persist():
        write_json(output / 'convergence-progress.json', progress)

    def evaluate(seed, source):
        check_budget()
        probe.seed = seed
        result = physical() if source == 'physical' else probe.objective()
        keys = ('proxy', 'hard', 'initial_state_drift') if source == 'physical' else ('proxy',)
        for key in keys:
            if result[key].shape != (probe.batch,) or not torch.isfinite(result[key]).all():
                raise FloatingPointError(f'Invalid {source} reference: {key}')
        if source == 'physical':
            if float(result['initial_state_drift'].max()) > 1e-5:
                raise ValueError('Physical reference failed fixed-state restore tolerance')
            progress['physical_rollouts'] += 1
        else:
            progress['wm_reference_rollouts'] += 1
        return {key: result[key].detach().cpu().double().numpy().copy() for key in keys}

    persist()
    try:
        for i, seed in enumerate(gradient_seeds):
            check_budget()
            probe.seed = seed
            pair = estimate_pair(probe)
            for m, method in enumerate(METHODS):
                moments[method].add(pair[method])
                projections[m, i] = (direction_vectors @ pair[method].double()).cpu().numpy()
                norms.append(dict(method=method, seed=seed, squared_l2=float(pair[method].double().square().sum())))
                if i + 1 in counts:
                    variance.append(dict(method=method, **moments[method].summary()))
            costs.append(dict(seed=seed, bptt_seconds=pair['bptt_seconds'], ppo_seconds=pair['ppo_seconds'],
                              imagined_steps_per_estimator=pair['imagined_steps_per_estimator']))
            del pair
            progress['gradient_draws'] = i + 1
            persist()
        torch.save({method: value.mean.cpu() for method, value in moments.items()}, output / 'final-means.pt')
        np.savez(output / 'projections.npz', projections=projections, gradient_seeds=gradient_seeds)
        save_csv(output / 'estimator-costs.csv', costs)
        save_csv(output / 'vector-variance.csv', variance)
        save_csv(output / 'gradient-norms.csv', norms)
        raw = dict(projections=projections, epsilons=np.array(epsilons), counts=np.array(counts))
        for source in ('physical', 'wm'):
            baselines = [evaluate(seed, source) for seed in reference_seeds]
            repeated = [baselines[0], *[evaluate(reference_seeds[0], source) for _ in range(repeats - 1)]]
            for key in (('proxy', 'hard', 'initial_state_drift') if source == 'physical' else ('proxy',)):
                name = 'wm' if source == 'wm' else 'physical_' + key
                raw[name + '_baseline'] = np.stack([b[key] for b in baselines])
                raw[name + '_repeats'] = np.stack([b[key] for b in repeated])
                raw[name] = np.empty((directions, len(epsilons), 2, len(reference_seeds), probe.batch))
        reference_started = time.perf_counter()
        displacements = []
        for d, vector in enumerate(direction_vectors):
            for e, epsilon in enumerate(epsilons):
                for sign_index, sign in enumerate((1, -1)):
                    with displaced_parameters(probe.policy, (sign * epsilon * vector).to(before.dtype)):
                        actual = (parameter_vector(probe.policy) - before).double()
                        relative_rounding = float((actual - sign * epsilon * vector).norm()) / epsilon
                        if relative_rounding > .02:
                            raise ValueError('FP32 parameter displacement differs from requested direction by more than 2% L2')
                        displacements.append(dict(direction=d, epsilon=epsilon, sign=sign,
                            actual_l2=float(actual.norm()), actual_direction_projection=float(actual @ vector),
                            relative_rounding_l2=relative_rounding))
                        for n, seed in enumerate(reference_seeds):
                            simulated, imagined = evaluate(seed, 'physical'), evaluate(seed, 'wm')
                            for key in ('proxy', 'hard', 'initial_state_drift'):
                                raw['physical_' + key][d, e, sign_index, n] = simulated[key]
                            raw['wm'][d, e, sign_index, n] = imagined['proxy']
                persist()
            # Partial data remains identifiable if an external budget ends the run.
            np.savez(output / 'partial-reference.npz', **{k: v[:d+1] for k, v in raw.items()
                      if k in ('physical_proxy', 'physical_hard', 'physical_initial_state_drift', 'wm')})
        raw.update(reference_seeds=np.array(reference_seeds), gradient_seeds=np.array(gradient_seeds))
        np.savez(output / 'raw-reference.npz', **raw)
        save_csv(output / 'parameter-displacements.csv', displacements)
        analyze(raw, output)
        result = dict(status='complete', **design, variance=variance,
            physical_rollouts=progress['physical_rollouts'], wm_reference_rollouts=progress['wm_reference_rollouts'],
            reference_phase_seconds=time.perf_counter() - reference_started,
            raw_sha256=sha256(output / 'raw-reference.npz'), means_sha256=sha256(output / 'final-means.pt'))
        write_json(output / 'convergence-summary.json', result)
        progress['status'] = 'complete'
        return result
    except BaseException as exc:
        progress.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        probe.seed = original_seed
        persist()
        if not torch.equal(parameter_vector(probe.policy), before):
            raise RuntimeError('Probe changed fixed policy parameters')
