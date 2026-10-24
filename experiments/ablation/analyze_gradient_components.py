"""Decompose one fixed policy gradient using the complete critic/action 2x2 grid.

This explains the WM gradient, not the physical Jacobian. Adam displacements
are nonlinear in the new gradient and are deliberately not decomposed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .analyze import save_csv
from .analyze_policy_fidelity import compare
from .common import write_json


def component_statistics(captures):
    """Keys are (no_critic, detach_action_reward); all four are required."""
    if set(captures) != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError('Require a complete 2x2 critic/direct-action gradient grid')
    full = captures[False, False]
    for (no_critic, detach), capture in captures.items():
        if capture['no_critic'] != no_critic or capture['detach_action_reward'] != detach:
            raise ValueError('Gradient capture labels differ from the requested grid')
        for key in ('parameter_layout', 'horizon', 'gamma', 'seed', 'provenance'):
            if capture[key] != full[key]:
                raise ValueError(f'Gradient captures differ in {key}')
    for no_critic in (False, True):
        torch.testing.assert_close(captures[no_critic, False]['objective'],
                                   captures[no_critic, True]['objective'], rtol=0, atol=0)
    gradients = {key: value['raw_ascent_gradient'].double() for key, value in captures.items()}
    if any(g.shape != gradients[False, False].shape or not torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError('Gradient shapes or finite values differ')
    total = gradients[False, False]
    norm = total.norm()
    if norm <= 0:
        raise ValueError('Cannot normalize a zero full-target gradient')
    state = gradients[True, True]
    direct = gradients[True, False] - state
    bootstrap = gradients[False, True] - state
    residual = total - state - direct - bootstrap
    relative_residual = float(residual.norm() / norm)
    if relative_residual > 2e-5:
        raise ValueError('2x2 gradients fail the additive objective identity')
    rows = []
    for name, vector in [('state_reward_and_soft_survival', state),
                         ('explicit_action_reward', direct), ('critic_bootstrap', bootstrap),
                         ('numerical_residual', residual)]:
        component_norm = vector.norm()
        projection = vector @ total / norm
        rows.append(dict(component=name, l2=float(component_norm),
                         cosine_with_full=float(projection / component_norm) if component_norm > 0 else None,
                         projection_on_full_unit=float(projection),
                         fraction_of_full_projection=float(projection / norm)))
    return dict(horizon=full['horizon'], gradient_dimension=total.numel(), full_gradient_l2=float(norm),
                additive_residual_relative_l2=relative_residual, components=rows,
                interpretation='WM gradient contributions at one fixed snapshot; not physical gradient accuracy',
                direct_action_scope='Only explicit reward action/previous-action arguments are detached; state-mediated paths remain',
                adam_scope='Historical Adam moments retain previous critic/reward effects; updates are not additive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probes', nargs=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    probes = compare(args.probes)
    if len({p['receipt']['args']['gradient_horizon'] for p in probes}) != 1:
        raise ValueError('Analyze one horizon at a time')
    captures = {}
    for probe in probes:
        config = probe['receipt']['args']
        captures[config['no_critic'], config['detach_action_reward']] = torch.load(
            Path(probe['path']) / 'gradients.pt', map_location='cpu', weights_only=False)
    result = component_statistics(captures)
    result['probes'] = [p['path'] for p in probes]
    result['all_tensors_verified'] = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / 'gradient-components.csv', result['components'])
    write_json(output / 'gradient-components.json', result)
    plot_components(result, output)


def plot_components(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['components'][:3]
    labels = ['State reward + soft survival', 'Explicit action reward', 'Critic bootstrap']
    colors = ['#4477AA', '#EE6677', '#228833']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for axis, field, title in zip(axes, ('l2', 'fraction_of_full_projection'),
                                  ('Component gradient L2', 'Signed share of full-gradient projection')):
        bars = axis.barh(labels, [row[field] for row in rows], color=colors)
        axis.invert_yaxis()
        axis.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
        axis.axvline(0, color='#555555', lw=.7)
        axis.set_xlabel(title)
        axis.margins(x=.25)
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"H={result['horizon']}: additive WM gradient at one fixed snapshot\n"
                 'Component norms do not add; this is not physical-gradient accuracy', fontsize=12)
    for extension in ('png', 'svg'):
        path = output / f'gradient-components.{extension}'
        fig.savefig(path, dpi=180)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
