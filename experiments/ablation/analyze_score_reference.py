"""Independent score-reference uncertainty and conditional gradient comparisons."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .analyze import save_csv
from .analyze_checkpoint_audit import batch_statistics
from .common import write_json


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
