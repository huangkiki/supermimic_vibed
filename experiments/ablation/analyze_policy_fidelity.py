"""Validate and compare completed policy-gradient probes without running physics.

All summaries describe one fixed checkpoint and paired replay minibatches. Neither
perturbation directions nor repeated evaluations count as independent training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import save_csv
from .common import sha256, write_json
from .fidelity_math import reference_quality
from .policy_fidelity import summarize_direction_scan



def mean_difference_reference(after, before):
    """FP64 reference plus a recorded-dtype reduction/subtraction roundoff bound.

    Saved per-start returns are the inputs, not recomputed trajectories. For n
    additions/division operations, gamma_n = n*u/(1-n*u), u = eps/2. Allow n+2
    operations per mean and one final subtraction. This bounds roundoff only;
    it is not a bound on simulator error, model bias or sampling uncertainty.
    """
    if (after.dtype not in (torch.float32, torch.float64) or before.dtype != after.dtype or after.shape != before.shape
            or after.ndim != 1 or after.numel() == 0 or not torch.isfinite(after).all()
            or not torch.isfinite(before).all()):
        raise ValueError('Expected paired finite FP32/FP64 per-start returns')
    unit = torch.finfo(after.dtype).eps / 2
    operations = after.numel() + 2
    if operations * unit >= 1:
        raise ValueError('Return batch exceeds the reduction-bound domain')
    a, b = after.double(), before.double()
    reference = float(a.mean() - b.mean())
    mean_error = operations * unit / (1 - operations * unit) * float(a.abs().mean() + b.abs().mean())
    bound = mean_error + unit * (abs(reference) + mean_error)
    if after.dtype == torch.float64:
        bound *= 2  # The FP64 audit reference itself also undergoes this rounding.
    return reference, bound


def validate_mean_difference(reported, after, before):
    reference, bound = mean_difference_reference(after, before)
    if not np.isfinite(reported) or abs(reported - reference) > bound:
        raise ValueError('Actual Adam improvement differs from saved physical rollouts beyond recorded-dtype roundoff')
    return reference, bound


def load_probe(path, *, verify_tensors=True):
    path = Path(path)
    receipt = json.loads((path / 'probe.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError(f'Incomplete policy-gradient probe: {path}')
    arguments = receipt['args']
    if arguments['gradient_horizon'] not in (1, 8, 32):
        raise ValueError('Unsupported policy-gradient horizon')
    with (path / 'directions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty policy-gradient measurements')
    boolean = ('sign_agrees', 'resolved')
    strings = ('direction', 'objective', 'interpretation')
    for row in rows:
        for key in boolean:
            if row[key] not in ('True', 'False'):
                raise ValueError(f'Invalid boolean: {key}')
            row[key] = row[key] == 'True'
        for key in row.keys() - set(boolean) - set(strings):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError(f'Nonfinite measurement: {key}')
    keys = [(r['direction'], r['objective'], r['epsilon']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate direction/objective/epsilon measurement')
    summary = json.loads((path / 'summary.json').read_text())
    directions = {r['direction'] for r in rows}
    requested = {'raw_gradient', 'clipped_gradient', 'historical_adam'}
    requested.update(f'random_{i}' for i in range(arguments['directions']))
    skipped = [item['direction'] for item in summary['skipped']]
    if (len(set(skipped)) != len(skipped) or directions.intersection(skipped)
            or directions.union(skipped) != requested):
        raise ValueError('Requested directions are missing or unaccounted for')
    expected = {(name, objective, epsilon) for name in directions
                for objective in ('proxy', 'hard') for epsilon in arguments['epsilons']}
    if set(keys) != expected or any(r['horizon'] != arguments['gradient_horizon'] for r in rows):
        raise ValueError('Incomplete or mixed policy-gradient measurement grid')
    for name in directions:
        for objective in ('proxy', 'hard'):
            measured = sorted((r for r in rows if r['direction'] == name and r['objective'] == objective),
                              key=lambda r: r['epsilon'])
            noise = measured[0]['baseline_repeat_noise']
            if noise < 0 or any(r['baseline_repeat_noise'] != noise for r in measured):
                raise ValueError('Inconsistent numerical noise')
            delta = np.array([r['positive_return'] - r['negative_return'] for r in measured])
            derivative = delta / (2 * np.array([r['epsilon'] for r in measured]))
            # Saved tensor arithmetic is float32, whereas CSV subtraction here is float64.
            if not np.allclose(derivative, [r['numerical_derivative'] for r in measured], rtol=1e-5, atol=1e-8):
                raise ValueError('Finite difference differs from saved paired returns')
            stable = reference_quality(derivative, delta, noise)
            if not np.array_equal(stable, [r['resolved'] for r in measured]):
                raise ValueError('Numerical resolution flag differs from paired returns')
            for row in measured:
                if row['sign_agrees'] != bool(np.sign(row['predicted_derivative']) == np.sign(row['numerical_derivative'])):
                    raise ValueError('Direction sign flag differs from the measured derivatives')
                if not np.isclose(row['positive_improvement'], row['positive_return'] - row['baseline_return'], rtol=1e-5, atol=1e-8):
                    raise ValueError('Reported improvement differs from physical returns')
                expected_interpretation = ('matched_proxy_derivative' if objective == 'proxy'
                    and not arguments['detach_action_reward'] else 'update_direction_effectiveness_only')
                if row['interpretation'] != expected_interpretation:
                    raise ValueError('Physical effectiveness mislabeled as matching-target accuracy')
    recomputed = summarize_direction_scan(dict(rows=rows, actual_adam_improvement=summary['actual_adam_improvement'],
        skipped=summary['skipped'], independent_unit=summary['summaries'][0]['scope'],
        hard_return_interpretation=summary['hard_return_interpretation']))
    if recomputed != summary:
        raise ValueError('Direction summary differs from its measurements')
    norms, per_start = {}, []
    if verify_tensors:
        bank = path / 'policy-bank.pt'
        if sha256(bank) != receipt['prepared_bank_sha256']:
            raise ValueError('Prepared bank bytes differ from the run receipt')
        raw_bank = torch.load(bank, map_location='cpu', weights_only=False)
        capture = torch.load(path / 'gradients.pt', map_location='cpu', weights_only=False)
        rollout = torch.load(path / 'direction-rollouts.pt', map_location='cpu', weights_only=False)
        dimensions = sum(item['numel'] for item in capture['parameter_layout'])
        for key in ('raw_loss_gradient', 'norm_clipped_loss_gradient', 'clipped_loss_gradient',
                    'raw_ascent_gradient', 'clipped_ascent_gradient', 'adam_displacement'):
            vector = capture[key]
            if vector.shape != (dimensions,) or not torch.isfinite(vector).all():
                raise ValueError(f'Invalid saved parameter vector: {key}')
            norms[key + '_l2'] = float(vector.double().norm())
        torch.testing.assert_close(capture['raw_ascent_gradient'], -capture['raw_loss_gradient'], rtol=0, atol=0)
        torch.testing.assert_close(capture['clipped_ascent_gradient'], -capture['clipped_loss_gradient'], rtol=0, atol=0)
        torch.testing.assert_close(capture['clipped_loss_gradient'], capture['norm_clipped_loss_gradient'].clamp(-.1, .1), rtol=0, atol=0)
        raw_norm = float(capture['raw_loss_gradient'].double().norm())
        if not np.isclose(float(capture['raw_gradient_norm']), raw_norm, rtol=2e-5, atol=1e-7):
            raise ValueError('Recorded raw norm differs from saved gradient')
        scale = (1 / (capture['raw_gradient_norm'] + 1e-6)).clamp(max=1)
        torch.testing.assert_close(capture['norm_clipped_loss_gradient'],
                                   capture['raw_loss_gradient'] * scale, rtol=2e-6, atol=1e-8)
        torch.testing.assert_close(capture['loss'], -capture['objective'].mean(), rtol=2e-6, atol=1e-7)
        if set(rollout['unit_directions']) != directions:
            raise ValueError('Saved parameter directions do not match the measured grid')
        selected = dict(raw_gradient=capture['raw_ascent_gradient'],
                        clipped_gradient=capture['clipped_ascent_gradient'],
                        historical_adam=capture['adam_displacement'])
        for name, direction in rollout['unit_directions'].items():
            if (direction.shape != (dimensions,) or not torch.isfinite(direction).all()
                    or not np.isclose(float(direction.double().norm()), 1., rtol=2e-5, atol=1e-7)):
                raise ValueError('Saved parameter direction is not a finite unit vector')
            if name in selected:
                # A 9M-element FP32 CPU norm can lose substantially more
                # precision than the original CUDA reduction. Reconstruct the
                # unit direction in FP64 rather than relaxing the acceptance.
                vector = selected[name].double()
                torch.testing.assert_close(direction.double(), vector / vector.norm(), rtol=2e-5, atol=1e-8)
            predicted = float(capture['raw_ascent_gradient'].double() @ direction.double())
            if any(not np.isclose(row['predicted_derivative'], predicted, rtol=2e-5, atol=1e-7)
                   for row in rows if row['direction'] == name):
                raise ValueError('Predicted derivative differs from saved gradient dot direction')
        if capture['horizon'] != arguments['gradient_horizon'] or capture['no_critic'] != arguments['no_critic'] or capture['detach_action_reward'] != arguments['detach_action_reward']:
            raise ValueError('Gradient tensor and reported target differ')
        if rollout['rows'] != rows:
            raise ValueError('CSV differs from retained raw rollout measurements')
        for row in rows:
            index = arguments['epsilons'].index(row['epsilon'])
            plus, minus, plus_norm, minus_norm = rollout['perturbations'][row['direction']][index]
            objective = row['objective']
            for label, measured in (('positive_return', plus[objective].mean()),
                                    ('negative_return', minus[objective].mean()),
                                    ('baseline_return', rollout['baselines'][0][objective].mean())):
                if not np.isclose(float(measured), row[label], rtol=2e-6, atol=1e-7):
                    raise ValueError('CSV return differs from the raw physical rollout')
            if plus_norm != row['positive_parameter_displacement_l2'] or minus_norm != row['negative_parameter_displacement_l2']:
                raise ValueError('Recorded parameter displacement differs from the raw scan')
        physical = [*rollout['baselines'], rollout['actual_adam']]
        physical.extend(item for pairs in rollout['perturbations'].values() for pair in pairs for item in pair[:2])
        for item in physical:
            drift = item.get('initial_state_drift')
            if drift is None or not torch.isfinite(drift).all() or float(drift.max()) > 1e-5:
                raise ValueError('Missing or invalid raw physical-state restore evidence')
        indices = raw_bank.get('source_sample_indices', arguments['sample_indices'])
        names = raw_bank.get('motion_names', ['unspecified'] * len(indices))
        for objective in ('proxy', 'hard'):
            before = rollout['baselines'][0][objective]
            after = rollout['actual_adam'][objective]
            for i, index in enumerate(indices):
                per_start.append(dict(source_sample_index=index, motion=names[i], objective=objective,
                                      baseline_return=float(before[i]), updated_return=float(after[i]),
                                      actual_adam_improvement=float(after[i] - before[i])))
        for key, improvement in summary['actual_adam_improvement'].items():
            reference, bound = validate_mean_difference(improvement, rollout['actual_adam'][key], rollout['baselines'][0][key])
            norms['adam_improvement_fp64_' + key] = reference
            norms['adam_improvement_roundoff_bound_' + key] = bound
        raw, adam = capture['raw_ascent_gradient'].double(), capture['adam_displacement'].double()
        norms['raw_adam_cosine'] = float((raw @ adam) / (raw.norm() * adam.norm())) if raw.norm() * adam.norm() > 0 else None
    return dict(path=str(path), receipt=receipt, rows=rows, summary=summary, norms=norms,
                tensors_verified=verify_tensors, per_start=per_start)


def compare(paths, *, verify_tensors=True):
    probes = [load_probe(path, verify_tensors=verify_tensors) for path in paths]
    if not probes:
        raise ValueError('At least one completed probe required')
    cases = [(p['receipt']['args']['gradient_horizon'], p['receipt']['args']['no_critic'],
              p['receipt']['args']['detach_action_reward']) for p in probes]
    if len(set(cases)) != len(cases):
        raise ValueError('Repeated horizon/ablation case is not an independent observation')
    first = probes[0]['receipt']
    for probe in probes[1:]:
        current = probe['receipt']
        for key in ('checkpoint_sha256', 'precision', 'reset_protocol'):
            if current[key] != first[key]:
                raise ValueError(f'Cannot pool different {key}; analyze separately')
        for key in ('sample_indices', 'probe_seed', 'epsilons'):
            if current['args'][key] != first['args'][key]:
                raise ValueError(f'Cannot pool different fixed sampling: {key}')
        if current['manifest']['gpu_uuid'] != first['manifest']['gpu_uuid']:
            raise ValueError('Cannot pool different physical GPUs')
        if verify_tensors:
            original = torch.load(Path(probes[0]['path']) / 'policy-bank.pt', map_location='cpu', weights_only=False)
            candidate = torch.load(Path(probe['path']) / 'policy-bank.pt', map_location='cpu', weights_only=False)
            for key in ('state', 'frames', 'xi', 'previous_action'):
                torch.testing.assert_close(original[key], candidate[key], rtol=0, atol=0)
            for key in ('s', 'f', 'xi', 'a_prev'):
                torch.testing.assert_close(original['replay_history'][key], candidate['replay_history'][key], rtol=0, atol=0)
    return probes



def plot_comparison(probes, output, epsilon):
    """Export separate matching-target derivative and physical-improvement figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    horizons = sorted({p['receipt']['args']['gradient_horizon'] for p in probes})
    colors = {False: '#276FBF', True: '#D55E00'}
    critic_levels = sorted({p['receipt']['args']['no_critic'] for p in probes})
    fig, axes = plt.subplots(len(critic_levels), 2, figsize=(11, 3.6 * len(critic_levels) + .5), squeeze=False)
    for row_index, no_critic in enumerate(critic_levels):
        for column, objective in enumerate(('proxy', 'hard')):
            axis = axes[row_index, column]
            for detach in (False, True):
                group = sorted((p for p in probes if p['receipt']['args']['no_critic'] == no_critic
                                and p['receipt']['args']['detach_action_reward'] == detach),
                               key=lambda p: p['receipt']['args']['gradient_horizon'])
                if group:
                    axis.plot([horizons.index(p['receipt']['args']['gradient_horizon']) for p in group],
                              [p['summary']['actual_adam_improvement'][objective] for p in group],
                              marker='s' if detach else 'o', linestyle='--' if detach else '-', color=colors[detach], label='Direct action gradient off' if detach else 'Full reward gradient')
            axis.axhline(0, color='#555555', lw=.8)
            axis.set_xticks(range(len(horizons)), horizons)
            axis.set_xlabel('Rollout horizon H')
            axis.set_ylabel('Change in discounted return')
            axis.set_title(('No critic | ' if no_critic else 'Critic bootstrap | ') +
                           ('Matching soft proxy' if objective == 'proxy' else 'Hard physical return'))
            axis.grid(alpha=.2)
    handles = [Line2D([0], [0], color=colors[detach], marker='o', label=label)
               for detach, label in ((False, 'Full reward gradient'), (True, 'Direct action gradient off'))]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(.5, .015))
    fig.suptitle('Actual historical-Adam step: proxy improvement and physical improvement', fontsize=13)
    samples = len(probes[0]['receipt']['args']['sample_indices'])
    fig.text(.5, .075, f'One checkpoint, {samples} fixed starts; paired diagnostics, no training-seed confidence interval', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.95))
    for ext in ('png','svg'):
        fig.savefig(output / f'actual-adam-improvement.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(critic_levels), len(horizons), figsize=(max(11, 4.2 * len(horizons)), 3.6 * len(critic_levels)), squeeze=False)
    for row_index, no_critic in enumerate(critic_levels):
        for column, horizon in enumerate(horizons):
            axis = axes[row_index, column]
            matches = [p for p in probes if p['receipt']['args']['gradient_horizon'] == horizon
                       and p['receipt']['args']['no_critic'] == no_critic
                       and not p['receipt']['args']['detach_action_reward']]
            if not matches:
                axis.set_visible(False)
                continue
            rows = [r for r in matches[0]['rows'] if r['objective'] == 'proxy' and r['epsilon'] == epsilon]
            if not rows:
                raise ValueError('Requested figure epsilon is absent from a probe')
            low = min([0.] + [r[k] for r in rows for k in ('predicted_derivative','numerical_derivative')])
            high = max([0.] + [r[k] for r in rows for k in ('predicted_derivative','numerical_derivative')])
            pad = max(.08 * (high - low), 1e-5)
            axis.plot([low-pad, high+pad], [low-pad, high+pad], '--', color='#777777', lw=1)
            for r in rows:
                color = '#D55E00' if r['direction'].startswith('random') else '#276FBF'
                axis.scatter(r['predicted_derivative'], r['numerical_derivative'], color=color,
                             marker='o' if r['resolved'] else 'x', s=45)
            axis.set_xlim(low-pad, high+pad); axis.set_ylim(low-pad, high+pad)
            axis.set_title(f"H={horizon}, {'no critic' if no_critic else 'critic'}; {sum(r['resolved'] for r in rows)}/{len(rows)} resolved")
            axis.set_xlabel('WM parameter-gradient dot direction')
            axis.set_ylabel('Physical central finite difference')
            axis.grid(alpha=.2)
    fig.suptitle(f'Matching soft-target directional derivatives, epsilon={epsilon:g}', fontsize=13)
    fig.text(.5,.02,'Blue: selected updates; orange: random directions. x = unresolved. Finite differences are numerical references.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ('png','svg'):
        fig.savefig(output / f'directional-reference.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probes', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--figure-epsilon', type=float, default=.01)
    args = parser.parse_args()
    probes = compare(args.probes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics, directions, per_start = [], [], []
    for probe in probes:
        receipt, config = probe['receipt'], probe['receipt']['args']
        identity = dict(horizon=config['gradient_horizon'], no_critic=config['no_critic'],
                        detach_action_reward=config['detach_action_reward'])
        metrics.append(dict(**identity, **probe['norms'],
            **{'actual_adam_' + key + '_improvement': value for key, value in probe['summary']['actual_adam_improvement'].items()},
            probe_main_seconds=receipt['seconds'], tensors_verified=probe['tensors_verified']))
        directions.extend({**identity, **row} for row in probe['rows'])
        per_start.extend({**identity, **row} for row in probe['per_start'])
    save_csv(output / 'updates.csv', metrics)
    save_csv(output / 'directions.csv', directions)
    save_csv(output / 'per-start-updates.csv', per_start)
    write_json(output / 'analysis.json', dict(probes=[p['path'] for p in probes], tensor_validation=True,
        checkpoint_sha256=probes[0]['receipt']['checkpoint_sha256'],
        scope='single fixed-policy paired diagnostics; no independent-training-seed confidence interval'))
    plot_comparison(probes, output, args.figure_epsilon)


if __name__ == '__main__':
    main()
