"""Verify raw matched-estimator artifacts and plot conditional sampling statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import save_csv
from .common import sha256, write_json
from .compare_policy_estimators import sample_statistics, validate_design
from .fidelity_math import reference_quality


def load_comparison(path):
    path = Path(path)
    receipt = json.loads((path/'probe.json').read_text())
    summary = json.loads((path/'comparison-summary.json').read_text())
    if receipt['status'] != 'complete' or summary['status'] != 'complete':
        raise ValueError('Incomplete matched-estimator diagnostic')
    args = receipt['args']
    validate_design(summary['gradient_seeds'], summary['reference_seeds'], args['epsilons'])
    if summary['gradient_seeds'] != args['estimator_seeds'] or summary['reference_seeds'] != args['reference_seeds']:
        raise ValueError('Noise groups differ from the requested experiment')
    if summary['action_std'] != args['action_std'] or summary['horizon'] != args['gradient_horizon']:
        raise ValueError('Target differs from the experiment receipt')
    if sha256(path/'policy-bank.pt') != receipt['prepared_bank_sha256']:
        raise ValueError('Fixed-state bank differs from its receipt')
    estimates = [torch.load(path/f'estimate-{seed}.pt',map_location='cpu',weights_only=False)
                 for seed in summary['gradient_seeds']]
    saved = torch.load(path/'mean-gradients.pt',map_location='cpu',weights_only=False)
    for record, seed in zip(estimates, summary['gradient_seeds']):
        if record['rollout']['seed'] != seed or record['rollout']['std'] != summary['action_std']:
            raise ValueError('Saved estimate belongs to another sampling distribution')
        if record['imagined_steps_per_estimator'] != summary['horizon']*summary['batch']:
            raise ValueError('Unequal rollout sample accounting')
    for method in ('bptt','ppo'):
        matrix = torch.stack([record[method] for record in estimates]).double()
        if matrix.ndim != 2 or not torch.isfinite(matrix).all():
            raise ValueError('Invalid raw gradient estimates')
        mean = matrix.mean(0)
        torch.testing.assert_close(saved['means'][method],mean,rtol=1e-12,atol=1e-12)
        direction = saved['unit_directions'][method]
        torch.testing.assert_close(direction.double(),mean/mean.norm(),rtol=2e-6,atol=1e-8)
        variance = next(item for item in summary['variance'] if item['method']==method)
        if not np.isclose(variance['variance_trace'],float(matrix.var(0,correction=1).sum()),rtol=1e-10,atol=1e-12):
            raise ValueError('Sampling variance differs from saved estimates')
    raw = torch.load(path/'physical-reference.pt',map_location='cpu',weights_only=False)
    if raw['reference_seeds'] != summary['reference_seeds'] or len(raw['baseline']) != len(summary['reference_seeds']):
        raise ValueError('Physical noise group changed')
    with (path/'comparison.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    if csv_rows != [{key:str(value) for key,value in row.items()} for row in raw['rows']]:
        raise ValueError('Comparison CSV differs from the raw scan')
    expected = {(method, objective, epsilon) for method in ('bptt','ppo')
                for objective in ('proxy','hard') for epsilon in args['epsilons']}
    if len(raw['rows']) != len(expected) or {(r['direction'],r['objective'],r['epsilon']) for r in raw['rows']} != expected:
        raise ValueError('Missing or repeated comparison cell')
    physical = [*raw['baseline'], *raw['numerical_repeats']]
    physical.extend(value for scans in raw['perturbations'].values() for scan in scans for key in ('plus','minus') for value in scan[key])
    for item in physical:
        for key in ('proxy','hard','initial_state_drift'):
            if item[key].shape != (summary['batch'],) or not torch.isfinite(item[key]).all():
                raise ValueError('Invalid physical reference tensor')
        if float(item['initial_state_drift'].max())>1e-5:
            raise ValueError('Physical restore drift exceeds tolerance')
    for method, scans in raw['perturbations'].items():
        if [scan['epsilon'] for scan in scans] != args['epsilons']:
            raise ValueError('Raw parameter-scale grid changed')
        for objective in ('proxy','hard'):
            noise = max(float((item[objective].mean()-raw['baseline'][0][objective].mean()).abs())
                        for item in raw['numerical_repeats'][1:])
            delta = [torch.tensor([float(p[objective].mean()-m[objective].mean())
                     for p,m in zip(scan['plus'],scan['minus'])],dtype=torch.float64) for scan in scans]
            derivatives = [d/(2*scan['epsilon']) for d,scan in zip(delta,scans)]
            resolved = reference_quality(np.array([float(d.mean()) for d in derivatives]),
                                         np.array([float(d.mean()) for d in delta]),noise)
            for index, scan in enumerate(scans):
                row = next(r for r in raw['rows'] if (r['direction'],r['objective'],r['epsilon'])==(method,objective,scan['epsilon']))
                improvements = torch.tensor([float(p[objective].mean()-b[objective].mean())
                    for p,b in zip(scan['plus'],raw['baseline'])],dtype=torch.float64)
                for label, values in (('derivative',derivatives[index]),('improvement',improvements)):
                    stats = sample_statistics(values)
                    for key,value in stats.items():
                        if not np.isclose(row[f'{label}_{key}'],value,rtol=1e-12,atol=1e-12):
                            raise ValueError('Sampling statistic differs from raw physical returns')
                if row['epsilon_noise_resolved'] != bool(resolved[index]) or row['numerical_repeat_noise'] != noise:
                    raise ValueError('Numerical sensitivity differs from the raw reference')
                for estimator in ('bptt','ppo'):
                    predicted = float(saved['means'][estimator]@saved['unit_directions'][method].double())
                    if not np.isclose(row[f'{estimator}_predicted_derivative'],predicted,rtol=1e-12,atol=1e-12):
                        raise ValueError('Directional prediction differs from the retained gradient')
    return dict(path=str(path),receipt=receipt,summary=summary,rows=raw['rows'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('probes',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--epsilon',type=float,default=.01)
    args = parser.parse_args()
    probes = sorted([load_comparison(path) for path in args.probes],key=lambda p:p['summary']['horizon'])
    first = probes[0]['receipt']
    if len({p['summary']['horizon'] for p in probes})!=len(probes):
        raise ValueError('Repeated horizons require separate analyses')
    for probe in probes[1:]:
        for key in ('checkpoint_sha256','precision'):
            if probe['receipt'][key]!=first[key]:
                raise ValueError(f'Cannot combine different {key}')
        for key in ('action_std','no_critic','sample_indices'):
            if probe['receipt']['args'][key]!=first['args'][key]:
                raise ValueError(f'Cannot combine different {key}')
        if probe['receipt']['manifest']['gpu_uuid']!=first['manifest']['gpu_uuid']:
            raise ValueError('Cannot combine different physical GPUs')
        original = torch.load(Path(probes[0]['path'])/'policy-bank.pt',map_location='cpu',weights_only=False)
        candidate = torch.load(Path(probe['path'])/'policy-bank.pt',map_location='cpu',weights_only=False)
        for key in ('state','frames','xi','previous_action'):
            torch.testing.assert_close(original[key],candidate[key],rtol=0,atol=0)
        for key in ('s','f','xi','a_prev'):
            torch.testing.assert_close(original['replay_history'][key],candidate['replay_history'][key],rtol=0,atol=0)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    save_csv(output/'comparison.csv',[dict(horizon=p['summary']['horizon'],**row) for p in probes for row in p['rows']])
    save_csv(output/'sampling-variance.csv',[dict(horizon=p['summary']['horizon'],**row) for p in probes for row in p['summary']['variance']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for column,objective in enumerate(('proxy','hard')):
        axis=axes[column]
        for offset,(method,color) in enumerate((('bptt','#276FBF'),('ppo','#D55E00'))):
            rows=[next(r for r in p['rows'] if r['direction']==method and r['objective']==objective and r['epsilon']==args.epsilon) for p in probes]
            positions=np.arange(len(probes))+(offset-.5)*.3
            axis.bar(positions,[r['improvement_mean'] for r in rows],width=.28,color=color,label=method.upper(),
                     yerr=[r['improvement_standard_error'] for r in rows],capsize=4)
        axis.axhline(0,color='#555555',lw=.8)
        axis.set_xticks(np.arange(len(probes)),[str(p['summary']['horizon']) for p in probes])
        axis.set_xlabel('Horizon H')
        axis.set_ylabel('Change in discounted return')
        axis.set_title('Soft + critic proxy' if objective=='proxy' else 'Hard physical return')
        axis.grid(axis='y',alpha=.2)
        axis.legend()
    fig.suptitle(f'Matched Gaussian policy: equal parameter step L2={args.epsilon:g}')
    fig.text(.5,.025,'Error bars: SE across held-out noise tapes; one checkpoint, two fixed starts; no training-seed interval',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.savefig(output/'equal-step-improvement.png',dpi=180)
    plt.close(fig)
    write_json(output/'validation.json',dict(status='passed',probes=[p['path'] for p in probes],
        tensor_validation=True,checkpoint_sha256=first['checkpoint_sha256'],
        scope='conditional sampling and local update diagnostics; no algorithm ranking'))


if __name__=='__main__':
    main()
