"""Recompute checkpoint audit statistics without native simulation dependencies."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from .analyze import save_csv
from .common import write_json
from .fidelity_math import reference_quality


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def mean_se(values):
    value = np.asarray(values, dtype=float)
    if len(value) < 2 or not np.isfinite(value).all():
        raise ValueError('At least two finite independent noise batches required')
    return float(value.mean()), float(value.std(ddof=1) / np.sqrt(len(value)))


def batch_statistics(records, keys, field):
    """Cluster by the actual independent noise seed, not individual directions."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        group = tuple(row[key] for key in keys)
        groups[group][int(row['seed'])].append(float(row[field]))
    out = []
    for group, seeds in sorted(groups.items()):
        batches = [np.mean(values) for values in seeds.values()]
        mean, se = mean_se(batches) if len(batches) >= 2 else (float(batches[0]), '')
        out.append(dict(zip(keys, group), mean=mean, standard_error=se, noise_batches=len(seeds),
                        observations=sum(map(len, seeds.values()))))
    return out


def analyze(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / 'audit-plan.json').read_text())
    summary = json.loads((root / 'audit-summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Only complete audit evidence can be analyzed')
    dirs = np.load(root / 'bias-directions.npy', allow_pickle=False)
    expected = (2*np.random.default_rng(plan['direction_seed']).integers(0, 2, dirs.shape)-1)/math.sqrt(dirs.shape[1])
    np.testing.assert_array_equal(dirs, expected)
    baselines = rows(root / 'physical-baselines.csv')
    predictions = rows(root / 'prediction.csv')
    updates = rows(root / 'physical-updates.csv')
    measured = rows(root / 'directional-returns.csv')
    draws = json.loads((root / 'gradient-draws.json').read_text())
    models = [Path(p).stem for p in plan['wm_checkpoints']]
    methods = ['ppo_sim', *[model+'_'+method for model in models for method in ('bptt', 'ppo')]]
    batch = max(int(row['sample']) for row in baselines)+1
    n_ref, n_grad, n_dir, n_eps = len(plan['reference_seeds']), len(plan['gradient_seeds']), len(dirs), len(plan['epsilons'])
    if (len(measured) != 2*n_dir*n_ref*batch*(n_eps+len(models))
            or len(updates) != len(methods)*n_ref*batch*n_eps
            or len(predictions) != len(models)*n_ref*batch*4
            or len(draws) != len(methods)*n_grad or len(baselines) != n_ref*batch):
        raise ValueError('Raw observation count differs from registered complete design')
    projection = {}
    for method in methods:
        selected = [row for row in draws if row['method'] == method]
        if [row['seed'] for row in selected] != plan['gradient_seeds']:
            raise ValueError('Gradient seed order/count changed')
        bias = np.array([row['bias'] for row in selected])
        projection[method] = bias @ dirs.T
        np.testing.assert_allclose(projection[method], [row['projections'] for row in selected], rtol=1e-12, atol=1e-12)
    ref_index = {seed: i for i, seed in enumerate(plan['reference_seeds'])}
    eps_index = {epsilon: i for i, epsilon in enumerate(plan['epsilons'])}
    references = {model: np.full((n_dir, n_eps, 2, n_ref, batch), np.nan) for model in ['physical', *models]}
    seen = set()
    for row in measured:
        model = row['model']
        index = (int(row['direction']), eps_index[float(row['epsilon'])], 0 if int(row['sign']) == 1 else 1,
                 ref_index[int(row['seed'])], int(row['sample']))
        if (model, index) in seen:
            raise ValueError('Duplicate numerical reference observation')
        seen.add((model, index))
        references[model][index] = float(row['proxy'])
        if not 0 <= float(row['relative_rounding']) <= .02:
            raise ValueError('Unqualified parameter displacement')
    primary = eps_index[plan['primary_epsilon']]
    for model, values in references.items():
        needed = values if model == 'physical' else values[:, primary]
        if not np.isfinite(needed).all():
            raise ValueError('Missing/nonfinite numerical reference')
    baseline0 = np.mean([float(row['proxy']) for row in baselines if int(row['seed']) == plan['reference_seeds'][0]])
    repeat = json.loads((root / 'baseline-repeats.json').read_text())
    if len(repeat) != plan['repeats']-1:
        raise ValueError('Incomplete baseline repeat evidence')
    noise = max(abs(np.mean(row['proxy'])-baseline0) for row in repeat)
    values = references['physical'].mean(axis=-1)
    delta = values[:, :, 0] - values[:, :, 1]
    physical = delta / (2*np.array(plan['epsilons'])[None, :, None])
    ref_mean, ref_se = physical.mean(-1), physical.std(-1, ddof=1)/math.sqrt(n_ref)
    resolved = np.array([reference_quality(mean, difference.mean(-1), noise)
                         for mean, difference in zip(ref_mean, delta)]) & (abs(ref_mean) > 2*ref_se)
    directional, comparisons = [], []
    for method in methods:
        estimate = projection[method].mean(0)
        se = projection[method].std(0, ddof=1)/math.sqrt(n_grad)
        for e, epsilon in enumerate(plan['epsilons']):
            keep = resolved[:, e]
            error = estimate-ref_mean[:, e]
            for d in range(n_dir):
                directional.append(dict(method=method, direction=d, epsilon=epsilon, estimate=estimate[d],
                    estimate_se=se[d], reference=ref_mean[d, e], reference_se=ref_se[d, e],
                    error=error[d], resolved=bool(keep[d]), sign_agrees=bool(np.sign(estimate[d])==np.sign(ref_mean[d,e])),
                    error_exceeds_two_combined_se=bool(abs(error[d]) > 2*np.hypot(se[d], ref_se[d,e]))))
            comparisons.append(dict(method=method, epsilon=epsilon, total=n_dir, resolved=int(keep.sum()),
                rmse_all=float(np.sqrt(np.mean(error**2))), reference_se_rms=float(np.sqrt(np.mean(ref_se[:,e]**2))),
                estimate_se_rms=float(np.sqrt(np.mean(se**2))),
                rmse_resolved=float(np.sqrt(np.mean(error[keep]**2))) if keep.any() else '',
                sign_agreement=float(np.mean(np.sign(estimate[keep])==np.sign(ref_mean[keep,e]))) if keep.any() else '',
                pearson_resolved=float(np.corrcoef(estimate[keep],ref_mean[keep,e])[0,1]) if keep.sum() >= 3 and np.std(estimate[keep])>0 else ''))
    internal = []
    for model in models:
        value = references[model][:, primary].mean(-1)
        numerical = (value[:,0]-value[:,1])/(2*plan['primary_epsilon'])
        for method in ('bptt', 'ppo'):
            error = projection[model+'_'+method].mean(0)-numerical.mean(-1)
            internal.append(dict(checkpoint=model, method=method, rmse=float(np.sqrt(np.mean(error**2))),
                                 numerical_se_rms=float(np.sqrt(np.mean((numerical.std(-1,ddof=1)/np.sqrt(n_ref))**2)))))
    prediction_summary = batch_statistics(predictions, ('checkpoint','horizon'), 'scaled_mse')
    update_summary = batch_statistics(updates, ('method','epsilon'), 'proxy_improvement')
    hard_summary = batch_statistics(updates, ('method','epsilon'), 'hard_improvement')
    grouped_prediction = batch_statistics(predictions, ('checkpoint','horizon','contact_group'), 'scaled_mse')
    grouped_updates = batch_statistics(updates, ('method','epsilon','contact_group'), 'hard_improvement')
    baseline_summary = batch_statistics(baselines, (), 'proxy')
    failure_rows = [{**row, 'failure_rate': int(row['failed'] == 'True')} for row in baselines]
    baseline_summary[0]['tracking_failure_rate'] = batch_statistics(failure_rows, (), 'failure_rate')[0]['mean']
    for filename, data in (('directional-comparison', directional), ('physical-accuracy', comparisons),
            ('within-wm-accuracy', internal), ('prediction-summary', prediction_summary), ('update-summary', update_summary),
            ('hard-update-summary', hard_summary), ('prediction-by-contact', grouped_prediction),
            ('hard-update-by-contact', grouped_updates), ('baseline-summary', baseline_summary)):
        save_csv(output / (filename+'.csv'), data)
    validation = dict(status='passed', reference_scope='output-bias subspace', full_parameters=summary['parameter_count'],
                      directions=n_dir, gradient_noise_batches=n_grad, reference_noise_batches=n_ref, batch=batch,
                      original_starts=batch//plan['replicas'], primary_epsilon=plan['primary_epsilon'],
                      baseline_repeat_noise=noise, resolved_by_epsilon=dict(zip(map(str,plan['epsilons']),map(int,resolved.sum(0)))),
                      contact_counts=dict((group,sum(row['contact_group']==group for row in baselines))
                                          for group in sorted(set(row['contact_group'] for row in baselines))),
                      independent_training_seeds=1, statement='Noise SEs do not quantify training or state-distribution uncertainty')
    write_json(output / 'validation.json', validation)
    return dict(predictions=prediction_summary, accuracy=comparisons, updates=update_summary,
                hard_updates=hard_summary, internal=internal, validation=validation)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    primary = result['validation']['primary_epsilon']
    models = sorted({row['checkpoint'] for row in result['predictions']})
    iterations = [int(name.split('_')[1]) for name in models]
    fig, ax = plt.subplots(figsize=(8,4.6), constrained_layout=True)
    for h in ('1','8','16','32'):
        chosen = [next(row for row in result['predictions'] if row['checkpoint']==model and row['horizon']==h) for model in models]
        ax.errorbar(iterations, [r['mean'] for r in chosen], yerr=[2*r['standard_error'] for r in chosen], marker='o', label=f'H={h}')
    ax.set(xlabel='WM checkpoint iteration (policy fixed at 4000)', ylabel='Scaled open-loop state MSE', yscale='log',
           title='Same physical states and actions; error bars = 2 noise SE')
    ax.legend(ncol=4);fig.savefig(output/'prediction.png',dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.8), constrained_layout=True)
    for kind, style in [('bptt','o-'),('ppo','s--')]:
        chosen = [next(row for row in result['accuracy'] if row['method']==model+'_'+kind and row['epsilon']==primary) for model in models]
        axes[0].plot(iterations,[row['rmse_all'] for row in chosen],style,label='BPTT' if kind=='bptt' else 'PPO-WM')
        update = [next(row for row in result['hard_updates'] if row['method']==model+'_'+kind and float(row['epsilon'])==primary) for model in models]
        axes[1].errorbar(iterations,[row['mean'] for row in update],yerr=[2*row['standard_error'] for row in update],fmt=style,label='BPTT' if kind=='bptt' else 'PPO-WM')
    physical = next(row for row in result['accuracy'] if row['method']=='ppo_sim' and row['epsilon']==primary)
    axes[0].axhline(physical['rmse_all'],color='k',ls=':',label='PPO-Sim')
    axes[0].axhline(physical['reference_se_rms'],color='grey',ls='--',label='Physical reference SE RMS')
    update = next(row for row in result['hard_updates'] if row['method']=='ppo_sim' and float(row['epsilon'])==primary)
    axes[1].axhline(update['mean'],color='k',ls=':',label='PPO-Sim')
    axes[1].axhspan(update['mean']-2*update['standard_error'],update['mean']+2*update['standard_error'],color='grey',alpha=.15)
    axes[1].axhline(0,color='grey',lw=.7)
    axes[0].set(ylabel='RMSE to physical finite differences (all directions)',
                title=f"Output-bias subspace: {physical['resolved']}/{physical['total']} resolved")
    axes[1].set(ylabel='Paired physical hard-return improvement',title=f'Full-vector unit step {primary:g}; 2 noise SE')
    for ax in axes: ax.set_xlabel('WM iteration');ax.legend(fontsize=8)
    fig.savefig(output/'gradient-and-update.png',dpi=180);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result=analyze(args.input,args.output)
    plot(result,args.output)
    print(json.dumps(result['validation'],indent=2))


if __name__ == '__main__':
    main()
