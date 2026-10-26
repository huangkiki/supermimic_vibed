"""Physical effectiveness at matched Gaussian KL on a separate calibration tape."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch

from .analyze import save_csv
from .common import sha256, write_json
from .policy_fidelity import displaced_parameters, parameter_vector


@torch.no_grad()
def calibrate_direction(actor, observations, goals, direction, target, check_budget, *, max_step=.1, return_means=False):
    """Solve for output RMS on fixed inputs, preserving original batch shapes.

    This is KL on a declared calibration state distribution, not trajectory KL.
    Equal-variance Gaussian KL averaged across states equals A*RMS^2/(2*sigma^2).
    """
    if target <= 0 or not math.isfinite(target) or not torch.isclose(direction.norm(),direction.new_tensor(1.),atol=1e-5):
        raise ValueError('Require a finite positive RMS and a unit parameter direction')
    baseline=torch.stack([actor.mean_policy(obs,goal) for obs,goal in zip(observations,goals)])
    before=parameter_vector(actor)

    def evaluate(step):
        check_budget()
        with displaced_parameters(actor,step*direction):
            shifted=torch.stack([actor.mean_policy(obs,goal) for obs,goal in zip(observations,goals)])
            actual=parameter_vector(actor)-before
        return float((shifted-baseline).square().mean().sqrt()),float((actual-step*direction).norm()/step),shifted

    lower,upper=0.,min(.0001,max_step)
    rms,rounding,shifted=evaluate(upper)
    while rms < target and upper < max_step:
        lower,upper=upper,min(2*upper,max_step)
        rms,rounding,shifted=evaluate(upper)
    if rms < target:
        raise ValueError('Calibration target exceeds the declared parameter-step cap')
    for _ in range(24):
        step=(lower+upper)/2
        rms,rounding,shifted=evaluate(step)
        if abs(rms-target) <= .01*target:
            break
        if rms < target:
            lower=step
        else:
            upper=step
    if abs(rms-target) > .01*target or rounding > .02:
        raise ValueError('Unresolved action-RMS calibration or parameter quantization')
    torch.testing.assert_close(parameter_vector(actor),before,rtol=0,atol=0)
    result=dict(parameter_step=step,requested_action_rms=target,actual_action_rms=rms,
                gaussian_kl=baseline.shape[-1]*rms*rms/(2*actor.std**2),parameter_rounding_relative_l2=rounding)
    if return_means:
        result['means']=dict(baseline=baseline.cpu(),shifted=shifted.cpu())
    return result


def run_updates(probe,physical,output,plan,check_budget):
    output=Path(output);parent=Path(plan['update_only_parent'])
    write_json(output/'audit-plan.json',plan)
    summary=json.loads((parent/'audit-summary.json').read_text())
    parent_plan=json.loads((parent/'audit-plan.json').read_text())
    parent_receipt=json.loads((parent/'probe.json').read_text())
    if summary['status']!='complete' or parent_receipt['status']!='complete':
        raise ValueError('Complete parent experiment required')
    if (parent_receipt['checkpoint_sha256']!=physical.bank['collector_sha256']
            or parent_receipt['args']['action_std']!=probe.policy.std):
        raise ValueError('Policy checkpoint or Gaussian policy differs from parent')
    for key in ('gradient_seeds','reference_seeds','replicas','wm_checkpoints'):
        if plan[key]!=parent_plan[key]:
            raise ValueError(f'Calibration parent design changed: {key}')
    if plan['calibration_seed'] in set(plan['gradient_seeds'])|set(plan['reference_seeds']):
        raise ValueError('Calibration noise must be independent of estimator and evaluation tapes')
    if sha256(parent/'gradient-means.pt')!=summary['gradient_means_sha256']:
        raise ValueError('Parent full gradient means changed')
    if sha256(output/'policy-bank.pt')!=parent_receipt['prepared_bank_sha256']:
        # torch.save serialization is not a tensor identity: compare actual bank
        # values below instead of assuming identical binary container bytes.
        saved=torch.load(parent/'policy-bank.pt',map_location=probe.device,weights_only=False)
        for key in ('state','frames','xi','previous_action'):
            torch.testing.assert_close(physical.bank[key],saved[key],rtol=0,atol=0)
        for key in physical.bank['replay_history']:
            torch.testing.assert_close(physical.bank['replay_history'][key],saved['replay_history'][key],rtol=0,atol=0)
    vectors=torch.load(parent/'gradient-means.pt',map_location=probe.device,weights_only=True)
    methods=['ckpt_004000_bptt','ckpt_004000_ppo','ppo_sim']
    targets=plan['target_action_rms']
    if targets != sorted(set(targets)) or any(not 0 < x <= .003 for x in targets):
        raise ValueError('Require increasing small action RMS targets up to .003')
    started=time.perf_counter();steps=0;rows=[];calibrations=[];baselines=[];timings=[];mean_tapes=[]
    physical.record_contacts=False

    def rollout(score=False):
        nonlocal steps
        check_budget()
        torch.cuda.synchronize(probe.device);start=time.perf_counter()
        result=physical.score_rollout() if score else physical()
        torch.cuda.synchronize(probe.device)
        timings.append(dict(seconds=time.perf_counter()-start,seed=probe.seed,score=score))
        steps+=probe.batch*probe.horizon
        return result

    probe.seed=plan['calibration_seed']
    calibration=rollout(score=True)
    torch.save({key:calibration[key].cpu() for key in ('obs','goal','action','objective')},output/'calibration-tape.pt')
    for seed in plan['reference_seeds']:
        probe.seed=seed;result=rollout()
        baselines.append({key:result[key].cpu() for key in ('proxy','hard','actions','initial_state_drift')})
    torch.save(baselines,output/'baseline-tapes.pt')
    from .physical_score_reference import reference_estimates
    reference=reference_estimates(probe,physical,output,plan,check_budget,candidate_means=vectors)
    steps+=reference['simulator_steps']
    for method in methods:
        direction=vectors[method]/vectors[method].norm()
        for target in targets:
            info=calibrate_direction(probe.policy,calibration['obs'],calibration['goal'],direction,target,check_budget,return_means=True)
            mean_tapes.append(dict(method=method,target=target,**info.pop('means')))
            torch.save(mean_tapes,output/'calibration-means.pt')
            calibrations.append(dict(method=method,**info))
            for seed,baseline in zip(plan['reference_seeds'],baselines):
                probe.seed=seed
                with displaced_parameters(probe.policy,info['parameter_step']*direction):
                    result=rollout()
                for n in range(probe.batch):
                    rows.append(dict(method=method,target_action_rms=target,seed=seed,sample=n,
                        proxy_improvement=float(result['proxy'][n].cpu()-baseline['proxy'][n]),
                        hard_improvement=float(result['hard'][n].cpu()-baseline['hard'][n]),
                        baseline_proxy=float(baseline['proxy'][n]),baseline_hard=float(baseline['hard'][n]),
                        updated_proxy=float(result['proxy'][n]),updated_hard=float(result['hard'][n]),
                        initial_action_rms=float((result['actions'][0,n].cpu()-baseline['actions'][0,n]).square().mean().sqrt())))
            save_csv(output/'calibration.csv',calibrations);save_csv(output/'updates.csv',rows)
            print(f'CALIBRATED {method} action RMS {target:g} complete',flush=True)
    save_csv(output/'timings.csv',timings)
    torch.save({name:vectors[name].cpu() for name in methods},output/'selected-means.pt')
    result=dict(status='complete',simulator_steps=steps,imagined_steps=reference['imagined_steps'],seconds=time.perf_counter()-started,
        physical_reference=reference,
        parent=str(parent),parent_summary_sha256=sha256(parent/'audit-summary.json'),
        parent_mean_sha256=summary['gradient_means_sha256'],calibration_seed=plan['calibration_seed'],
        calibration_tape_sha256=sha256(output/'calibration-tape.pt'),baseline_tapes_sha256=sha256(output/'baseline-tapes.pt'),
        calibration_means_sha256=sha256(output/'calibration-means.pt'),
        target_action_rms=targets,methods=methods,reference_seeds=plan['reference_seeds'],
        scope='selected full-vector update effects at matched calibration-distribution Gaussian KL; no gradient accuracy claim')
    write_json(output/'calibrated-summary.json',result)
    return result
