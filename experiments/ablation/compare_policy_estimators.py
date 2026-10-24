"""Run a bounded, fixed-policy comparison of two stochastic gradient estimators."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .analyze import save_csv
from .common import write_json
from .fidelity_math import reference_quality
from .policy_estimators import estimate_pair
from .policy_fidelity import displaced_parameters, parameter_vector


def validate_design(gradient_seeds, reference_seeds, epsilons):
    for seeds in (gradient_seeds, reference_seeds):
        if len(seeds) < 2 or len(seeds) > 32 or len(set(seeds)) != len(seeds):
            raise ValueError('Require 2..32 distinct seeds in each sampling group')
    if set(gradient_seeds).intersection(reference_seeds):
        raise ValueError('Gradient and physical-reference noise seeds must be disjoint')
    if len(epsilons) < 2 or list(epsilons) != sorted(set(epsilons)) or any(not math.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError('Require increasing positive finite parameter perturbations')


def sample_statistics(values):
    """Noise-tape sampling uncertainty, not numerical or training-seed uncertainty."""
    values = values.double()
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError('Finite independent noise-tape measurements required')
    return dict(mean=float(values.mean()), sd=float(values.std()),
                standard_error=float(values.std()/math.sqrt(len(values))), noise_tapes=len(values))


def run_comparison(probe, physical, output, *, gradient_seeds, reference_seeds, epsilons,
                   repeats=3, check_budget=lambda: None):
    """Equal rollout samples and equal unit-L2 parameter steps, on held-out noise.

    This is the first on-policy PPO surrogate gradient, not multi-epoch PPO
    training. Historical Adam is intentionally not reused for this auxiliary
    equal-displacement comparison; original Adam diagnostics remain separate.
    """
    validate_design(gradient_seeds, reference_seeds, epsilons)
    if repeats < 2:
        raise ValueError('At least two identical numerical baseline repeats required')
    output = Path(output)
    estimates = {'bptt': [], 'ppo': []}
    costs = []
    initial_seed = probe.seed
    try:
        for seed in gradient_seeds:
            check_budget()
            probe.seed = seed
            pair = estimate_pair(probe)
            # Preserve raw estimates and detached rollouts; avoid a GPU-resident
            # stack of every full policy gradient during subsequent physics.
            saved = {key: value.cpu() if torch.is_tensor(value) else value for key, value in pair.items() if key != 'rollout'}
            saved['rollout'] = {key: value.cpu() if torch.is_tensor(value) else value for key, value in pair['rollout'].items()}
            torch.save(saved, output / f'estimate-{seed}.pt')
            for method in estimates:
                estimates[method].append(saved[method])
            costs.append(dict(seed=seed, bptt_seconds=pair['bptt_seconds'], ppo_seconds=pair['ppo_seconds'],
                              imagined_steps_per_estimator=pair['imagined_steps_per_estimator']))
            del pair, saved
        means, directions, variance = {}, {}, []
        for method, gradients in estimates.items():
            matrix = torch.stack(gradients).double()
            mean = matrix.mean(0)
            norm = mean.norm()
            if norm <= 0 or not torch.isfinite(norm):
                raise ValueError('Cannot assess an empty mean gradient direction')
            trace = matrix.var(0, correction=1).sum()
            means[method] = mean
            directions[method] = (mean/norm).to(device=probe.device, dtype=next(probe.policy.parameters()).dtype)
            variance.append(dict(method=method, mean_gradient_l2=float(norm), variance_trace=float(trace),
                mean_estimate_standard_error_l2=float((trace/len(matrix)).sqrt()),
                relative_variance=float(trace/norm.square()), noise_tapes=len(matrix)))
        cosine = float(means['bptt'] @ means['ppo']/(means['bptt'].norm()*means['ppo'].norm()))
        torch.save(dict(means=means, unit_directions={key: value.cpu() for key, value in directions.items()}),
                   output / 'mean-gradients.pt')
        def evaluate(seed):
            check_budget()
            probe.seed = seed
            result = physical()
            for key in ('proxy', 'hard', 'initial_state_drift'):
                if result[key].shape != (probe.batch,) or not torch.isfinite(result[key]).all():
                    raise FloatingPointError(f'Invalid physical reference: {key}')
            if float(result['initial_state_drift'].max()) > 1e-5:
                raise ValueError('Physical reference failed the fixed-state restore tolerance')
            return {key: result[key].detach().cpu().clone() for key in ('proxy','hard','initial_state_drift')}
        baseline = [evaluate(seed) for seed in reference_seeds]
        numerical_repeats = [baseline[0], *[evaluate(reference_seeds[0]) for _ in range(repeats-1)]]
        numerical_noise = {key: max(float((r[key].mean()-baseline[0][key].mean()).abs()) for r in numerical_repeats[1:])
                           for key in ('proxy','hard')}
        raw, rows = {}, []
        before = parameter_vector(probe.policy)
        for method, direction in directions.items():
            scans = []
            for epsilon in epsilons:
                with displaced_parameters(probe.policy, epsilon * direction):
                    plus_norm = float((parameter_vector(probe.policy)-before).norm())
                    plus = [evaluate(seed) for seed in reference_seeds]
                with displaced_parameters(probe.policy, -epsilon * direction):
                    minus_norm = float((parameter_vector(probe.policy)-before).norm())
                    minus = [evaluate(seed) for seed in reference_seeds]
                scans.append(dict(epsilon=epsilon, plus=plus, minus=minus,
                                  positive_displacement_l2=plus_norm, negative_displacement_l2=minus_norm))
            raw[method] = scans
            for objective in ('proxy','hard'):
                differences = [torch.tensor([float(p[objective].mean()-m[objective].mean())
                                for p,m in zip(scan['plus'],scan['minus'])],dtype=torch.float64) for scan in scans]
                derivatives = [delta/(2*epsilon) for delta,epsilon in zip(differences,epsilons)]
                resolved = reference_quality(np.array([float(v.mean()) for v in derivatives]),
                    np.array([float(v.mean()) for v in differences]), numerical_noise[objective])
                for i, scan in enumerate(scans):
                    derivative_stats = sample_statistics(derivatives[i])
                    improvements = torch.tensor([float(p[objective].mean()-b[objective].mean())
                        for p,b in zip(scan['plus'],baseline)],dtype=torch.float64)
                    improvement_stats = sample_statistics(improvements)
                    rows.append(dict(direction=method, objective=objective, epsilon=scan['epsilon'],
                        **{'derivative_'+key:value for key,value in derivative_stats.items()},
                        **{'improvement_'+key:value for key,value in improvement_stats.items()},
                        bptt_predicted_derivative=float(means['bptt']@direction.cpu().double()),
                        ppo_predicted_derivative=float(means['ppo']@direction.cpu().double()),
                        epsilon_noise_resolved=bool(resolved[i]), numerical_repeat_noise=numerical_noise[objective],
                        **{key:scan[key] for key in ('positive_displacement_l2','negative_displacement_l2')},
                        interpretation='matched expected soft target; finite-sample numerical reference'
                            if objective=='proxy' else 'physical effectiveness; different hard target'))
        torch.save(dict(baseline=baseline, numerical_repeats=numerical_repeats, perturbations=raw,
                        reference_seeds=reference_seeds, rows=rows), output/'physical-reference.pt')
        save_csv(output/'estimator-costs.csv',costs)
        save_csv(output/'sampling-variance.csv',variance)
        save_csv(output/'comparison.csv',rows)
        summary = dict(status='complete', gradient_seeds=gradient_seeds, reference_seeds=reference_seeds,
            horizon=probe.horizon, batch=probe.batch, action_std=probe.policy.std,
            no_critic=probe.no_critic, mean_gradient_cosine=cosine,
            independent_unit='noise tapes conditional on one fixed checkpoint and start minibatch',
            numerical_reference='paired finite differences of held-out Gaussian-noise mean; not an analytic Jacobian',
            uncertainty='SE across noise tapes; report separately from numerical repeats and training seeds',
            comparison='first on-policy PPO gradient (lambda=1, no normalization) vs reparameterized BPTT',
            budget='equal imagined trajectories per estimate and equal L2 parameter displacements; no Adam',
            gradient_samples_per_estimator=len(gradient_seeds)*probe.batch,
            physical_noise_tapes=len(reference_seeds), variance=variance, costs=costs)
        write_json(output/'comparison-summary.json',summary)
        return summary
    finally:
        probe.seed = initial_seed
