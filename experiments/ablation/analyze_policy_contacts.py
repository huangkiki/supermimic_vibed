"""Describe predeclared past-contact cohorts without treating them as causal strata.

Each cohort supplies ONE gradient of its batch mean. Future contacts and per-start
Adam improvements are observations, not independent parameter-gradient estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analyze import save_csv
from .analyze_policy_fidelity import load_probe
from .common import sha256, write_json
from .fidelity_math import body_contact_signatures, contact_groups, window_contact_groups
from .policy_fidelity import validate_starts


def checked_contacts(signatures, steps, batch, geom_body_ids):
    if len(signatures) != steps or any(len(row) != batch for row in signatures):
        raise ValueError('Incomplete time/sample contact layout')
    if not geom_body_ids or any(not isinstance(i, int) or i < 0 for i in geom_body_ids):
        raise ValueError('Invalid geometry/body mapping')
    for row in signatures:
        for pairs in row:
            for pair in pairs:
                if len(pair) != 2 or any(not isinstance(g, int) or g < 0 or g >= len(geom_body_ids) for g in pair):
                    raise ValueError('Invalid native geometry contact pair')
    return body_contact_signatures(signatures, geom_body_ids)


def contact_evidence(bank, rollout, mapping, *, group, indices, motions, horizon):
    """Keep past-only selection, symmetric initial neighborhood and future window separate."""
    if bank.get('source_sample_indices') != indices or bank.get('motion_names') != motions:
        raise ValueError('Selected starts differ from the predeclared cohort')
    batch = len(indices)
    prefix = checked_contacts(bank['prefix_contacts'], len(bank['prefix_actions']), batch, mapping)
    if not prefix:
        raise ValueError('Contact history is missing')
    past = contact_groups(prefix)[-1]
    if any(label != group for label in past):
        raise ValueError('Past-contact cohort differs from its predeclared label')
    records = [*rollout['baselines'], rollout['actual_adam']]
    records.extend(item for pairs in rollout['perturbations'].values() for pair in pairs for item in pair[:2])
    for record in records:
        checked_contacts(record.get('geometry_contacts', []), horizon, batch, mapping)
    future = body_contact_signatures(rollout['baselines'][0]['geometry_contacts'], mapping)
    labels = contact_groups(prefix + future)
    # Prefix contacts describe states after prefix steps; its last entry IS s0.
    initial = labels[len(prefix) - 1]
    window = window_contact_groups(labels, len(prefix), horizon)
    return [dict(source_sample_index=index, motion=motions[i], past_contact_group=past[i],
                 initial_symmetric_group=initial[i], future_window_group=window[i],
                 future_contact_switches=sum((prefix + future)[j][i] != (prefix + future)[j - 1][i]
                                             for j in range(len(prefix), len(prefix) + horizon)))
            for i, index in enumerate(indices)]


def analyze(root, plan):
    root = Path(root)
    prepared = root / 'history64/policy-bank.pt'
    parent = torch.load(prepared, map_location='cpu', weights_only=False)
    history_receipt = json.loads((root / 'history64/probe.json').read_text())
    if (history_receipt['status'] != 'complete' or sha256(prepared) != history_receipt['prepared_bank_sha256']
            or parent.get('parent_bank_sha256') != plan['source_bank_sha256']
            or parent.get('bank_kind') != 'derived_prefix_replay_endpoint'):
        raise ValueError('Derived-bank provenance is missing or different')
    mapping = json.loads((root / 'history64/contact-map.json').read_text())
    probes, contacts, metrics, directions, updates, per_start = [], [], [], [], [], []
    first = None
    for group, indices in plan['cases'].items():
        path = root / group
        probe = load_probe(path)
        receipt, args = probe['receipt'], probe['receipt']['args']
        if (receipt['checkpoint_sha256'] != plan['checkpoint_sha256']
                or receipt['bank_sha256'] != sha256(prepared) or args['sample_indices'] != indices
                or args['gradient_horizon'] != plan['horizon'] or args['probe_seed'] != plan['probe_seed']
                or args['epsilons'] != plan['epsilons'] or args['directions'] != plan['random_directions']
                or args['repeats'] != plan['repeats'] or args['no_critic'] or args['detach_action_reward']
                or receipt['precision'] != dict(tf32=False, amp='None', compiled=False)):
            raise ValueError('Contact probe does not match the declared experiment')
        if first is not None:
            for key in ('manifest', 'source', 'reset_protocol'):
                if receipt[key] != first[key]:
                    raise ValueError(f'Contact cohorts differ in {key}')
        first = receipt
        if json.loads((path / 'contact-map.json').read_text()) != mapping:
            raise ValueError('Contact geometry/body mapping changed')
        bank = torch.load(path / 'policy-bank.pt', map_location='cpu', weights_only=False)
        for key in ('state', 'legacy_state', 'frames', 'xi', 'previous_action'):
            torch.testing.assert_close(bank[key], parent[key][indices], rtol=0, atol=0)
        for key in ('s', 'f', 'xi', 'a_prev'):
            torch.testing.assert_close(bank['replay_history'][key], parent['replay_history'][key][indices], rtol=0, atol=0)
        starts = dict(s=bank['state'], f=bank['frames'], xi=bank['xi'], a_prev=bank['previous_action'])
        validate_starts(starts, bank['replay_history'], len(indices), bank['replay_history']['f'].shape[1])
        rollout = torch.load(path / 'direction-rollouts.pt', map_location='cpu', weights_only=False)
        observed = contact_evidence(bank, rollout, mapping['geom_body_ids'], group=group, indices=indices,
                                    motions=plan['motion_names'], horizon=plan['horizon'])
        contacts.extend(dict(cohort=group, **row) for row in observed)
        directions.extend(dict(cohort=group, **row) for row in probe['rows'])
        metrics.extend(dict(cohort=group, **row) for row in probe['summary']['summaries'])
        updates.append(dict(cohort=group, **probe['norms'], **probe['summary']['actual_adam_improvement']))
        per_start.extend(dict(cohort=group, **row) for row in probe['per_start'])
        probes.append(probe)
    return dict(probes=probes, contacts=contacts, metrics=metrics, directions=directions, updates=updates,
                per_start=per_start, prepared_bank_sha256=sha256(prepared))


def plots(result, output, epsilon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, probe in enumerate(result['probes']):
        name = Path(probe['path']).name
        for j, target in enumerate(('proxy', 'hard')):
            axis = axes[i, j]
            rows = [r for r in probe['rows'] if r['objective'] == target and r['epsilon'] == epsilon
                    and r['direction'].startswith('random_')]
            for resolved in (True, False):
                subset = [r for r in rows if r['resolved'] == resolved]
                axis.scatter([r['predicted_derivative'] for r in subset], [r['numerical_derivative'] for r in subset],
                             marker='o' if resolved else 'x', label='resolved' if resolved else 'unresolved')
            limit = max(abs(r[key]) for r in rows for key in ('predicted_derivative', 'numerical_derivative')) * 1.15
            axis.plot([-limit, limit], [-limit, limit], '--', color='#888888', lw=1)
            axis.set_xlim(-limit, limit); axis.set_ylim(-limit, limit)
            axis.set_title(name.replace('_', ' ') + ' | ' + target)
            axis.set_xlabel('WM gradient dot random direction')
            axis.set_ylabel('Physical central finite difference')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle(f'H32, 8 starts per cohort; 8 random directions; epsilon={epsilon:g}')
    fig.text(.5, .015, 'Past-only groups; future switches reported separately. Hard return measures effectiveness, not matched-target accuracy.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'random-directions.{ext}', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, probe in enumerate(result['probes']):
        values = [r['actual_adam_improvement'] for r in probe['per_start'] if r['objective'] == 'hard']
        ax.scatter([i] * len(values), values, marker='o', s=30, color='#276FBF', alpha=.7)
        ax.scatter(i, probe['summary']['actual_adam_improvement']['hard'], marker='D', s=85, color='#D55E00', zorder=3)
    ax.axhline(0, lw=1, color='#888888')
    ax.set_xticks([0, 1], ['Past stable contact', 'Past switch nearby'])
    ax.set_ylabel('Change in discounted hard physical return')
    ax.set_title('Historical Adam: one batch update per cohort')
    fig.text(.5, .02, 'Blue: fixed starts; orange: batch mean. No independent-gradient or training-seed confidence interval.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'historical-adam.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    result = analyze(args.root, plan)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('contacts', 'metrics', 'directions', 'updates', 'per_start'):
        save_csv(output / f'{name}.csv', result[name])
    write_json(output / 'analysis.json', dict(tensor_validation=True, plan=plan,
        prepared_bank_sha256=result['prepared_bank_sha256'],
        scope='two batch-mean gradients at different fixed states; no causal contact effect; no training-seed inference'))
    plots(result, output, .01)


if __name__ == '__main__':
    main()
