"""Verify and plot independent-direction convergence from compact native records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import save_csv
from .common import sha256, write_json
from .estimator_convergence import METHODS, analyze, validate_design


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def paired_decomposition(raw):
    """Separate estimator disagreement from WM/physics mismatch using paired SE."""
    wm = (raw['wm'][:, :, 0].mean(-1) - raw['wm'][:, :, 1].mean(-1)) / (2*raw['epsilons'][None, :, None])
    physics = (raw['physical_proxy'][:, :, 0].mean(-1) - raw['physical_proxy'][:, :, 1].mean(-1)) / (2*raw['epsilons'][None, :, None])
    model_gap = wm - physics
    rows = []
    for size in raw['counts']:
        estimator_gap = raw['projections'][1, :size] - raw['projections'][0, :size]
        for d in range(wm.shape[0]):
            for e, epsilon in enumerate(raw['epsilons']):
                rows.append(dict(samples=int(size), direction=d, epsilon=float(epsilon),
                    ppo_minus_bptt=float(estimator_gap[:, d].mean()),
                    ppo_minus_bptt_se=float(estimator_gap[:, d].std(ddof=1)/np.sqrt(size)),
                    wm_minus_physics=float(model_gap[d, e].mean()),
                    wm_minus_physics_se=float(model_gap[d, e].std(ddof=1)/np.sqrt(wm.shape[-1])),
                    interpretation='Paired Monte Carlo SE; numerical scale/repeat screens must be checked separately'))
    return rows


def validate(path, *, means_path=None):
    path = Path(path)
    summary = json.loads((path / 'convergence-summary.json').read_text())
    receipt = json.loads((path / 'probe.json').read_text())
    if summary['status'] != 'complete' or receipt['status'] != 'complete':
        raise ValueError('Require a completed native experiment')
    validate_design(summary['gradient_seeds'], summary['reference_seeds'], summary['counts'],
                    summary['epsilons'], summary['directions'], summary['direction_seed'])
    for key, arg in (('gradient_seeds', 'estimator_seeds'), ('reference_seeds', 'reference_seeds'),
                     ('counts', 'convergence_counts'), ('directions', 'directions'),
                     ('direction_seed', 'direction_seed'), ('epsilons', 'epsilons'),
                     ('horizon', 'gradient_horizon'), ('action_std', 'action_std')):
        if summary[key] != receipt['args'][arg]:
            raise ValueError(f'Receipt and experiment differ: {key}')
    if sha256(path / 'raw-reference.npz') != summary['raw_sha256']:
        raise ValueError('Raw reference hash mismatch')
    if sha256(path / 'directions.npz') != summary['directions_sha256']:
        raise ValueError('Independent direction hash mismatch')
    if sha256(path / 'policy-bank.pt') != receipt['prepared_bank_sha256']:
        raise ValueError('Fixed bank hash mismatch')
    with np.load(path / 'raw-reference.npz', allow_pickle=False) as archive:
        raw = dict(archive)
    for key, values in raw.items():
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite raw data: {key}')
    for key in ('epsilons', 'counts', 'gradient_seeds', 'reference_seeds'):
        np.testing.assert_array_equal(raw[key], summary[key])
    d, e, n, b = summary['directions'], len(summary['epsilons']), len(summary['reference_seeds']), summary['starts']
    if b != len(receipt['args']['sample_indices']) or receipt['precision'] != dict(tf32=False, amp='None', compiled=False):
        raise ValueError('Reference minibatch or precision differs from the native protocol')
    for source in ('wm', 'physical_proxy', 'physical_hard', 'physical_initial_state_drift'):
        if raw[source].shape != (d, e, 2, n, b) or raw[source + '_baseline'].shape != (n, b):
            raise ValueError('Missing reference cell')
        if raw[source + '_repeats'].shape != (receipt['args']['repeats'], b):
            raise ValueError('Missing numerical repeat')
    if max(raw[k].max() for k in raw if k.startswith('physical_initial_state_drift')) > 1e-5:
        raise ValueError('Physics initial state drift exceeded tolerance')
    count = len(summary['gradient_seeds'])
    if raw['projections'].shape != (2, count, d):
        raise ValueError('Missing estimator projection')
    expected_calls = n * (1 + 2*d*e) + receipt['args']['repeats'] - 1
    if summary['physical_rollouts'] != expected_calls or summary['wm_reference_rollouts'] != expected_calls:
        raise ValueError('Rollout accounting differs from complete grid')
    with np.load(path / 'directions.npz', allow_pickle=False) as archive:
        packed, parameters = archive['packed_signs'], int(archive['parameter_count'])
        if int(archive['seed']) != summary['direction_seed']:
            raise ValueError('Direction seed changed')
    signs = np.random.default_rng(summary['direction_seed']).integers(0, 2, size=(d, parameters), dtype=np.uint8)
    if parameters != sum(layer['numel'] for layer in summary['parameter_layout']):
        raise ValueError('Direction dimension differs from the policy layout')
    np.testing.assert_array_equal(packed, np.packbits(signs, axis=-1))
    costs = read_csv(path / 'estimator-costs.csv')
    if [int(c['seed']) for c in costs] != summary['gradient_seeds']:
        raise ValueError('Cost sample identities changed')
    for cost in costs:
        if int(cost['imagined_steps_per_estimator']) != summary['horizon']*b:
            raise ValueError('Unequal estimator sample accounting')
        if any(not np.isfinite(float(cost[m + '_seconds'])) or float(cost[m + '_seconds']) <= 0 for m in METHODS):
            raise ValueError('Invalid estimator duration')
    norms = read_csv(path / 'gradient-norms.csv')
    variance = read_csv(path / 'vector-variance.csv')
    for method in METHODS:
        records = [r for r in norms if r['method'] == method]
        if [int(r['seed']) for r in records] != summary['gradient_seeds']:
            raise ValueError('Gradient-norm sample identities changed')
        for size in summary['counts']:
            row = next(r for r in variance if r['method'] == method and int(r['samples']) == size)
            # Independent conservation check; final mean norm can additionally be checked against full vectors.
            trace = (sum(float(r['squared_l2']) for r in records[:size]) - size*float(row['mean_l2'])**2)/(size-1)
            np.testing.assert_allclose(trace, float(row['variance_trace']), rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(np.sqrt(trace/size), float(row['mean_standard_error_l2']), rtol=1e-9, atol=1e-10)
    displacements = read_csv(path / 'parameter-displacements.csv')
    expected = {(di, ep, sign) for di in range(d) for ep in summary['epsilons'] for sign in (-1, 1)}
    if len(displacements) != len(expected) or {(int(r['direction']), float(r['epsilon']), int(r['sign'])) for r in displacements} != expected:
        raise ValueError('Displacement grid incomplete')
    if any(not 0 <= float(r['relative_rounding_l2']) <= .02 for r in displacements):
        raise ValueError('Parameter rounding tolerance exceeded')
    if means_path is not None:
        if sha256(means_path) != summary['means_sha256']:
            raise ValueError('Full-vector means hash mismatch')
        means = torch.load(means_path, map_location='cpu', weights_only=True)
        for i, method in enumerate(METHODS):
            vector = means[method].double().numpy()
            if vector.shape != (parameters,) or not np.isfinite(vector).all():
                raise ValueError('Invalid full-vector mean')
            row = next(r for r in variance if r['method'] == method and int(r['samples']) == count)
            np.testing.assert_allclose(np.linalg.norm(vector), float(row['mean_l2']), rtol=1e-10)
            for direction, bits in enumerate(signs):
                prediction = ((bits.astype(float)*2-1)/np.sqrt(parameters)) @ vector
                np.testing.assert_allclose(prediction, raw['projections'][i, :, direction].mean(), rtol=1e-10, atol=1e-12)
    return raw, summary, costs, variance


def plot(output, rows, costs, summary, epsilon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(bptt='#2166ac', ppo='#d6604d')
    counts = summary['counts']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for axis, source, title in zip(axes[:2], ('wm', 'physical_proxy'), ('WM numerical reference', 'Physical soft + critic reference')):
        for method in METHODS:
            selected = [r for r in rows if r['method'] == method and r['source'] == source and r['epsilon'] == epsilon]
            axis.plot(counts, [r['all_direction_rmse'] for r in selected], 'o-', color=colors[method], label=method.upper())
        axis.axhline(selected[0]['reference_se_rms'], color='#555555', ls=':', label='Reference SE RMS')
        axis.set(xscale='log', yscale='log', xlabel='Noise samples per estimator', ylabel='Projected RMSE (all directions)', title=title)
        axis.set_xticks(counts, [str(c) for c in counts])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for method in METHODS:
        times = [sum(float(c[method + '_seconds']) for c in costs[:n]) for n in counts]
        selected = [r for r in rows if r['method'] == method and r['source'] == 'physical_proxy' and r['epsilon'] == epsilon]
        axes[2].plot(times, [r['all_direction_rmse'] for r in selected], 'o-', color=colors[method], label=method.upper())
        for t, row in zip(times, selected):
            axes[2].annotate(str(row['samples']), (t, row['all_direction_rmse']), xytext=(4, 5), textcoords='offset points', fontsize=8)
    axes[2].set(xscale='log', yscale='log', xlabel='Cumulative estimator operation seconds', ylabel='Projected physical-reference RMSE', title='Measured cost at sampled budgets')
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle(f'H{summary["horizon"]}: {summary["directions"]} preselected directions, epsilon={epsilon:g}')
    fig.text(.5, .025, 'One snapshot, two fixed starts. SE is conditional noise-tape uncertainty; dotted line is not an accuracy floor.\nOperation timing excludes setup, serialization and reference evaluations; allocated campaign cost is reported separately.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'convergence.{ext}', dpi=180)
    plt.close(fig)
    detailed = read_csv(output / 'directional-comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, source, title in zip(axes, ('wm', 'physical_proxy'), ('WM numerical reference', 'Physical soft + critic reference')):
        base = [r for r in detailed if r['source'] == source and r['method'] == 'bptt' and int(r['samples']) == counts[-1] and float(r['epsilon']) == epsilon]
        x = np.arange(len(base))
        axis.errorbar(x, [float(r['reference']) for r in base], yerr=[2*float(r['reference_se']) for r in base], fmt='ks', capsize=3, label='Reference ±2 SE')
        for offset, method in ((-.13, 'bptt'), (.13, 'ppo')):
            values = [r for r in detailed if r['source'] == source and r['method'] == method and int(r['samples']) == counts[-1] and float(r['epsilon']) == epsilon]
            axis.errorbar(x+offset, [float(r['prediction']) for r in values], yerr=[2*float(r['prediction_se']) for r in values], fmt='o', color=colors[method], capsize=3, label=method.upper()+' ±2 SE')
        axis.set_xticks(x, [str(i)+('' if r['resolved'] == 'True' else '*') for i, r in enumerate(base)])
        axis.axhline(0, color='#888888', lw=.7)
        axis.set(xlabel='Independent direction (* reference unresolved)', ylabel='Directional derivative', title=title)
        axis.grid(axis='y', alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(f'N={counts[-1]}, epsilon={epsilon:g}; all preselected directions retained')
    fig.text(.5, .025, 'Bars: ±2 conditional Monte Carlo SE, not a numerical-error bound or training-seed confidence interval.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'directional-estimates.{ext}', dpi=180)
    plt.close(fig)
    # A separate scale keeps PPO sampling noise from hiding WM/physics differences.
    fig, axis = plt.subplots(figsize=(9, 4.6))
    for offset, source, label, color in ((-.13, 'wm', 'WM finite difference', '#4d9221'), (.13, 'physical_proxy', 'Physical finite difference', '#222222')):
        values = [r for r in detailed if r['source'] == source and r['method'] == 'bptt' and int(r['samples']) == counts[-1] and float(r['epsilon']) == epsilon]
        x = np.arange(len(values))
        axis.errorbar(x+offset, [float(r['reference']) for r in values], yerr=[2*float(r['reference_se']) for r in values], fmt='s', color=color, capsize=3, label=label+' ±2 SE')
        if source == 'wm':
            axis.errorbar(x, [float(r['prediction']) for r in values], yerr=[2*float(r['prediction_se']) for r in values], fmt='o', color=colors['bptt'], capsize=3, label='BPTT ±2 SE')
    axis.set_xticks(x, [str(i) for i in x])
    axis.axhline(0, color='#888888', lw=.7)
    axis.set(xlabel='Independent parameter direction', ylabel='Directional derivative', title=f'WM target and physical target, N={counts[-1]}, epsilon={epsilon:g}')
    axis.grid(axis='y', alpha=.2)
    axis.legend(fontsize=9)
    fig.text(.5, .025, 'All directions shown. Bars describe noise tapes, not finite-difference error. See CSV for scale-resolution flags.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'wm-physics-reference.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--means', type=Path, help='Optional retained full-vector means for independent norm/projection checks')
    parser.add_argument('--epsilon', type=float, default=.01)
    args = parser.parse_args()
    raw, summary, costs, variance = validate(args.probe, means_path=args.means)
    if args.epsilon not in summary['epsilons']:
        raise ValueError('Plot epsilon must be in the frozen grid')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = analyze(raw, args.output)
    for name in ('convergence.csv', 'directional-comparison.csv'):
        if (args.probe / name).read_bytes() != (args.output / name).read_bytes():
            raise ValueError(f'Native and CPU analysis differ: {name}')
    accounting = []
    for size in summary['counts']:
        for method in METHODS:
            accounting.append(dict(method=method, samples=size, imagined_steps=size*summary['horizon']*summary['starts'],
                operation_seconds=sum(float(c[method + '_seconds']) for c in costs[:size])))
    save_csv(args.output / 'estimator-compute.csv', accounting)
    save_csv(args.output / 'paired-decomposition.csv', paired_decomposition(raw))
    plot(args.output, rows, costs, summary, args.epsilon)
    write_json(args.output / 'validation.json', dict(status='passed', raw_sha256=summary['raw_sha256'],
        directions_sha256=summary['directions_sha256'], means_sha256=summary['means_sha256'],
        full_vector_means_checked=args.means is not None, cuda_initialized=torch.cuda.is_initialized(),
        scope='Raw reference/projection CSV recomputation; norms conserve streaming variance. Full-vector validation only when --means supplied.'))


if __name__ == '__main__':
    main()
