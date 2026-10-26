"""Independent physical score estimates with a fixed zero-mean control variate.

E[score(WM)] = E[pathwise(WM)] when differentiation and expectation commute.
Thus score(physics) - score(WM) + pathwise(WM) targets the physical objective,
without claiming an analytic simulator Jacobian. The coefficient is fixed at1;
raw estimates and empirical variance are retained even if variance increases.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .common import sha256,write_json
from .estimator_convergence import VectorMoments
from .policy_estimators import estimate_pair,score_gradient


def reference_estimates(probe,physical,output,plan,check_budget,*,candidate_means):
    output=Path(output);parent=Path(plan['update_only_parent'])
    seeds=plan['physical_reference_seeds']
    if (len(seeds)!=64 or len(set(seeds))!=64 or set(seeds)&(set(plan['gradient_seeds'])|set(plan['reference_seeds'])|{plan['calibration_seed']})):
        raise ValueError('Require64 independent reference tapes, disjoint from all other tapes')
    directions=np.load(parent/'bias-directions.npy',allow_pickle=False)
    from .checkpoint_audit import bias_slice
    section=bias_slice(probe.policy)
    pairs=[('ckpt_004000_bptt','ppo_sim'),('ckpt_004000_bptt','ckpt_004000_ppo'),
           ('ckpt_004000_ppo','ppo_sim')]
    pairs.extend((f'ckpt_{i:06d}_bptt',f'ckpt_{i-1000:06d}_bptt') for i in (1000,2000,3000,4000))
    comparisons=[]
    for left,right in pairs:
        a,b=candidate_means[left].double(),candidate_means[right].double()
        comparisons.append((left,right,a-b,float(a@a-b@b)))
    moments={};draws=[];errors=[];started=time.perf_counter()
    for seed in seeds:
        check_budget();probe.seed=seed
        torch.cuda.synchronize(probe.device);start=time.perf_counter()
        rollout=physical.score_rollout()
        physical_gradient=score_gradient(probe.policy,rollout)['gradient']
        pair=estimate_pair(probe)
        estimates=dict(ppo_sim=physical_gradient,ppo_wm=pair['ppo'],bptt_wm=pair['bptt'],
                       corrected=physical_gradient.double()-pair['ppo'].double()+pair['bptt'].double())
        record=dict(seed=seed)
        for name,vector in estimates.items():
            if name not in moments:moments[name]=VectorMoments(vector)
            moments[name].add(vector)
            bias=vector[section].double().cpu().numpy()
            record[name]=dict(bias=bias.tolist(),projections=(directions@bias).tolist(),l2=float(vector.double().norm()))
        for left,right,difference,constant in comparisons:
            for name in ('ppo_sim','corrected'):
                errors.append(dict(seed=seed,left=left,right=right,reference=name,
                    squared_error_difference=constant-2*float(estimates[name].double()@difference)))
        torch.cuda.synchronize(probe.device)
        record['seconds']=time.perf_counter()-start;draws.append(record)
        write_json(output/'reference-gradient-draws.json',draws)
        write_json(output/'reference-error-differences.json',errors)
        if len(draws)%8==0:print(f'PHYSICAL SCORE REFERENCE {len(draws)}/64 complete',flush=True)
    torch.save({name:moment.mean.float().cpu() for name,moment in moments.items()},output/'reference-gradient-means.pt')
    result=dict(status='complete',seeds=seeds,statistics={name:moment.summary() for name,moment in moments.items()},
        mean_sha256=sha256(output/'reference-gradient-means.pt'),seconds=time.perf_counter()-started,
        simulator_steps=len(seeds)*probe.batch*probe.horizon,imagined_steps=2*len(seeds)*probe.batch*probe.horizon,
        identity='physical score - WM score + WM pathwise; coefficient fixed at1',
        scope='Monte Carlo physical gradient reference; unbiasedness assumes matched stochastic target and valid expectation differentiation; not an analytic truth',
        sampled_policy='same frozen Gaussian actor and critic; WM4000',bias_offset=section.start,bias_dimensions=section.stop-section.start)
    write_json(output/'reference-gradient-summary.json',result)
    return result
