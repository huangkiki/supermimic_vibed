"""Recompute calibrated-KL physical effects from stored means and raw returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import save_csv
from .analyze_checkpoint_audit import batch_statistics,rows
from .common import sha256,write_json


def analyze(root,output,*,parent=None):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'calibrated-summary.json').read_text())
    plan=json.loads((root/'audit-plan.json').read_text())
    receipt=json.loads((root/'probe.json').read_text())
    if summary['status']!='complete' or receipt['status']!='complete':raise ValueError('Complete calibrated run required')
    for name,key in [('calibration-tape.pt','calibration_tape_sha256'),('baseline-tapes.pt','baseline_tapes_sha256'),
                     ('calibration-means.pt','calibration_means_sha256')]:
        if sha256(root/name)!=summary[key]:raise ValueError('Calibrated tensor hash differs')
    tapes=torch.load(root/'calibration-means.pt',map_location='cpu',weights_only=False)
    baselines=torch.load(root/'baseline-tapes.pt',map_location='cpu',weights_only=False)
    calibration=rows(root/'calibration.csv');updates=rows(root/'updates.csv')
    rejected=json.loads((root/'calibration-rejections.json').read_text())
    batch=len(baselines[0]['proxy']);references=plan['reference_seeds'];targets=plan['target_action_rms'];methods=summary['methods']
    expected={(method,target) for method in methods for target in targets}
    rejected_keys={(r['method'],float(r['requested_action_rms'])) for r in rejected}
    accepted_keys={(r['method'],float(r['requested_action_rms'])) for r in calibration}
    if (accepted_keys|rejected_keys!=expected or accepted_keys&rejected_keys
            or len(calibration)!=len(accepted_keys) or len(rejected)!=len(rejected_keys)):
        raise ValueError('Calibration method/scale coverage differs')
    for row in rejected:
        if (abs(row['actual_action_rms']-row['requested_action_rms'])<=.01*row['requested_action_rms']
                and row['parameter_rounding_relative_l2']<=.02):
            raise ValueError('Rejected calibration does not violate registered criteria')
    means={(r['method'],r['target']):r for r in tapes}
    if set(means)!=accepted_keys or len(tapes)!=len(accepted_keys):raise ValueError('Calibration means coverage differs')
    sigma=receipt['args']['action_std']
    for row in calibration:
        target=float(row['requested_action_rms']);tape=means[(row['method'],target)]
        delta=tape['shifted']-tape['baseline'];rms=float(delta.square().mean().sqrt())
        kl=float(delta.double().square().sum(-1).mean()/(2*sigma**2))
        np.testing.assert_allclose(rms,float(row['actual_action_rms']),rtol=1e-6,atol=1e-9)
        np.testing.assert_allclose(kl,float(row['gaussian_kl']),rtol=1e-6)
        if abs(rms-target)>target*.01 or float(row['parameter_rounding_relative_l2'])>.02:
            raise ValueError('Unqualified calibration')
    seen=set()
    for row in updates:
        key=(row['method'],float(row['target_action_rms']),int(row['seed']),int(row['sample']))
        if key[:2] not in accepted_keys:raise ValueError('Update at an unresolved calibration')
        if key in seen:raise ValueError('Duplicate physical update')
        seen.add(key)
        baseline=baselines[references.index(key[2])]
        for objective in ('proxy','hard'):
            if float(row['baseline_'+objective])!=float(baseline[objective][key[3]]):
                raise ValueError('Physical update baseline changed')
            # Subtraction occurred in native FP32; recompute with the same units.
            expected_delta=float(torch.tensor(float(row['updated_'+objective]),dtype=torch.float32)
                                 -torch.tensor(float(row['baseline_'+objective]),dtype=torch.float32))
            if expected_delta!=float(row[objective+'_improvement']):raise ValueError('Physical improvement differs')
    if len(seen)!=len(accepted_keys)*len(references)*batch:raise ValueError('Incomplete physical update coverage')
    new_reference=0 if summary['reference_reused'] else len(plan['physical_reference_seeds'])
    if summary['simulator_steps']!=32*batch*(1+len(references)*(1+len(accepted_keys))+new_reference):
        raise ValueError('Calibrated simulator step ledger differs')
    result={kind:batch_statistics(updates,('method','target_action_rms'),kind+'_improvement') for kind in ('proxy','hard')}
    for kind,values in result.items():save_csv(output/(kind+'-improvement.csv'),values)
    save_csv(output/'calibration.csv',calibration)
    if rejected:save_csv(output/'unresolved-calibrations.csv',rejected)
    validation=dict(status='passed',raw_physical_updates=len(updates),batch=batch,noise_batches=len(references),
        calibration_points=len(calibration),unresolved_calibrations=len(rejected),reference_reused=summary['reference_reused'],
        max_rms_relative_error=max(abs(float(r['actual_action_rms'])/float(r['requested_action_rms'])-1) for r in calibration),
        simulator_steps=summary['simulator_steps'],independent_training_seeds=1,
        scope='paired physical effects at equal calibration-distribution Gaussian KL; noise SE only')
    write_json(output/'validation.json',validation)
    from .analyze_score_reference import analyze as analyze_reference
    analyze_reference(root,output,parent=parent)
    return result


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),constrained_layout=True)
    for ax,kind,title in zip(axes,('proxy','hard'),('Matched soft + critic target','Physical hard return')):
        for method,label in [('ckpt_004000_bptt','BPTT'),('ckpt_004000_ppo','PPO-WM'),('ppo_sim','PPO-Sim')]:
            chosen=sorted([r for r in result[kind] if r['method']==method],key=lambda r:float(r['target_action_rms']))
            ax.errorbar([float(r['target_action_rms']) for r in chosen],[r['mean'] for r in chosen],
                        yerr=[2*r['standard_error'] for r in chosen],marker='o',label=label)
        ax.axhline(0,color='grey',lw=.7);ax.set(title=title,xlabel='Calibrated mean action RMS',xscale='log',ylabel='Paired return improvement')
        ax.legend(fontsize=9)
    fig.suptitle('Fixed WM4000; equal Gaussian KL on an independent calibration tape; bars = 2 noise SE',fontsize=10)
    fig.savefig(Path(output)/'calibrated-improvement.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args();result=analyze(args.input,args.output,parent=args.parent);plot(result,args.output)
    from .analyze_score_reference import plot_comparisons
    plot_comparisons(args.output)


if __name__=='__main__':main()
