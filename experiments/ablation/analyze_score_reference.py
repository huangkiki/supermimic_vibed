"""Independent score-reference uncertainty and conditional gradient comparisons."""
from __future__ import annotations

import json
import math
import csv
from pathlib import Path

import numpy as np

from .analyze import save_csv
from .analyze_checkpoint_audit import batch_statistics
from .common import write_json


def simultaneous_intervals(values, *, seed=111001, draws=10000):
    """Seed-cluster bootstrap max-|t| intervals for all declared comparisons.

    The same resampled noise rows are used for all comparisons, preserving
    covariance. These are approximate simultaneous intervals, not distribution-
    free bounds or training-seed uncertainty.
    """
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or len(values)<16 or not np.isfinite(values).all():
        raise ValueError('Require at least16 finite independent reference batches')
    mean=values.mean(0);se=values.std(0,ddof=1)/math.sqrt(len(values));active=se>0
    if not active.any():return mean,mean,0.
    centered=values[:,active]-mean[active]
    indices=np.random.default_rng(seed).integers(0,len(values),size=(draws,len(values)))
    samples=centered[indices]
    sample_se=samples.std(1,ddof=1)/math.sqrt(len(values))
    if (sample_se<=0).any():raise ValueError('Degenerate studentized bootstrap sample')
    maximum=np.abs(samples.mean(1)/sample_se).max(1)
    critical=float(np.quantile(maximum,.95))
    return mean-critical*se,mean+critical*se,critical


def analyze(root,output,*,parent=None):
    root,output=Path(root),Path(output)
    summary=json.loads((root/'reference-gradient-summary.json').read_text())
    data=json.loads((root/'reference-gradient-draws.json').read_text())
    plan=json.loads((root/'audit-plan.json').read_text())
    if summary['status']!='complete' or [r['seed'] for r in data]!=plan['physical_reference_seeds']:
        raise ValueError('Reference seed coverage differs')
    arrays={method:np.array([row[method]['projections'] for row in data]) for method in ('ppo_sim','ppo_wm','bptt_wm','corrected')}
    for field in ('bias','projections'):
        values={method:np.array([row[method][field] for row in data]) for method in arrays}
        np.testing.assert_allclose(values['corrected'],values['ppo_sim']-values['ppo_wm']+values['bptt_wm'],rtol=1e-10,atol=1e-10)
    n=len(data);directions=[];stats=[]
    for method,values in arrays.items():
        mean,se=values.mean(0),values.std(0,ddof=1)/math.sqrt(n)
        for d in range(values.shape[1]):
            directions.append(dict(method=method,direction=d,mean=mean[d],standard_error=se[d],noise_batches=n))
        stats.append(dict(method=method,projected_variance_trace=float(values.var(0,ddof=1).sum()),
            projected_mean_se_l2=float(np.linalg.norm(se)),**summary['statistics'][method]))
    difference=arrays['ppo_sim']-arrays['ppo_wm'];gap=[]
    for d in range(difference.shape[1]):
        gap.append(dict(direction=d,physical_minus_wm_score_mean=float(difference[:,d].mean()),
                        standard_error=float(difference[:,d].std(ddof=1)/math.sqrt(n)),noise_batches=n))
    save_csv(output/'reference-directional.csv',directions);save_csv(output/'reference-statistics.csv',stats)
    save_csv(output/'paired-score-model-gap.csv',gap)
    errors=json.loads((root/'reference-error-differences.json').read_text())
    if len(errors)!=7*2*n:raise ValueError('Incomplete full-vector error-difference comparisons')
    seen={(r['seed'],r['left'],r['right'],r['reference']) for r in errors}
    if len(seen)!=len(errors):raise ValueError('Duplicate error-difference comparison')
    comparisons=batch_statistics(errors,('left','right','reference'),'squared_error_difference')
    for row in comparisons:
        row['interpretation']='negative: left finite gradient estimate has lower squared error; conditional on fixed candidate gradients'
        row['two_se_interval_excludes_zero']=abs(row['mean'])>2*row['standard_error']
    save_csv(output/'full-gradient-error-difference.csv',comparisons)
    keys=sorted({(row['left'],row['right']) for row in errors})
    lookup={(row['seed'],row['left'],row['right']):row['squared_error_difference']
            for row in errors if row['reference']=='corrected'}
    matrix=np.array([[lookup[(seed,left,right)] for left,right in keys] for seed in summary['seeds']])
    lower,upper,critical=simultaneous_intervals(matrix)
    intervals=[dict(left=left,right=right,mean=float(matrix[:,i].mean()),
                    lower_simultaneous_95=float(lower[i]),upper_simultaneous_95=float(upper[i]),
                    bootstrap_max_t_critical=critical,noise_batches=n,
                    interpretation='negative: left has lower conditional squared error; approximate simultaneous bootstrap')
               for i,(left,right) in enumerate(keys)]
    save_csv(output/'full-gradient-comparison-intervals.csv',intervals)
    if parent is not None:
        original=json.loads((Path(parent)/'gradient-draws.json').read_text())
        if set(r['seed'] for r in original)&set(summary['seeds']):raise ValueError('Reference is not independent of candidate gradient estimates')
        predictions={method:np.array([row['projections'] for row in original if row['method']==method])
                     for method in dict.fromkeys(row['method'] for row in original)}
        results=[]
        for method,values in predictions.items():
            for reference in ('corrected','ppo_sim'):
                residual=values.mean(0)-arrays[reference].mean(0)
                se=np.sqrt(values.var(0,ddof=1)/len(values)+arrays[reference].var(0,ddof=1)/n)
                for d in range(len(residual)):
                    results.append(dict(method=method,reference=reference,direction=d,estimate=values[:,d].mean(),
                        reference_mean=arrays[reference][:,d].mean(),difference=residual[d],combined_noise_se=se[d]))
        save_csv(output/'gradient-against-score-reference.csv',results)
    write_json(output/'score-reference-validation.json',dict(status='passed',noise_batches=n,
        projected_variance_ratio=float(arrays['corrected'].var(0,ddof=1).sum()/arrays['ppo_sim'].var(0,ddof=1).sum()),
        full_variance_ratio=summary['statistics']['corrected']['variance_trace']/summary['statistics']['ppo_sim']['variance_trace'],
        full_vector_comparisons=7,scope='relative squared errors conditional on primary gradient estimates; Monte Carlo SE, no analytic ground truth'))


def plot_comparisons(output):
    """Show all seven declared comparisons, with their simultaneous intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / 'full-gradient-comparison-intervals.csv').open() as stream:
        rows = {(row['left'], row['right']): row for row in csv.DictReader(stream)}
    panels = [
        ('WM training stages; fixed policy', [
            (f'ckpt_{step:06d}_bptt', f'ckpt_{step-1000:06d}_bptt',
             f'BPTT {step} minus {step-1000}')
            for step in (1000, 2000, 3000, 4000)
        ]),
        ('Final WM; 512 trajectories per candidate', [
            ('ckpt_004000_bptt', 'ckpt_004000_ppo', 'BPTT minus PPO-WM'),
            ('ckpt_004000_bptt', 'ppo_sim', 'BPTT minus PPO-Sim'),
            ('ckpt_004000_ppo', 'ppo_sim', 'PPO-WM minus PPO-Sim'),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for ax, (title, comparisons) in zip(axes, panels):
        for index, (left, right, label) in enumerate(comparisons):
            row = rows[left, right]
            mean = float(row['mean'])
            lower = float(row['lower_simultaneous_95'])
            upper = float(row['upper_simultaneous_95'])
            color = '#b45309' if lower > 0 else '#2563a6' if upper < 0 else '#64748b'
            ax.errorbar(mean, index, xerr=[[mean-lower], [upper-mean]],
                        fmt='o', color=color, capsize=4)
        ax.set_yticks(range(len(comparisons)), [row[2] for row in comparisons])
        ax.invert_yaxis()
        ax.axvline(0, color='#475569', linewidth=1)
        ax.grid(axis='x', alpha=.2)
        ax.set(title=title, xlabel='Squared gradient error: left minus right\nNegative = left closer to physical expected gradient')
    fig.suptitle('Full 9.1M-parameter estimates; independent physical score reference\n'
                 'Approximate simultaneous 95% intervals; fixed candidates, two starts, one training seed', fontsize=11)
    fig.savefig(output / 'full-gradient-comparisons.png', dpi=180)
    plt.close(fig)
