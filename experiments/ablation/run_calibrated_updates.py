"""Run one matched-KL follow-up using only the parent campaign's remaining cap."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .common import sha256,write_json
from .run_fidelity_cohort import execute_probe
from .run_suite import wait_for_idle_gpu


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent-campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    parent=args.parent_campaign.resolve();output=args.output.resolve()
    campaign=json.loads((parent/'campaign.json').read_text())
    if campaign['status']!='complete':raise ValueError('Parent must finish before any follow-up allocation')
    plan=campaign['plan'];consumed=campaign['allocated_gpu_hours']+plan.get('earlier_attempt_gpu_hours',0.)
    remaining=3600*(plan['hard_gpu_hours']-consumed)
    native=json.loads((parent/'experiment/probe.json').read_text())
    design=json.loads((parent/'experiment/audit-plan.json').read_text())
    design.update(update_only_parent=str(parent/'experiment'),calibration_seed=91001,target_action_rms=[.0003,.001,.003])
    timings=list(csv.DictReader((parent/'experiment/timings.csv').open()))
    physical=[float(row['seconds']) for row in timings if row['operation'].endswith('_physical_update')]
    rollout_count=1+len(design['reference_seeds'])*(1+3*len(design['target_action_rms']))
    # Explicit 240-second calibration/setup allowance plus measured rollout
    # duration with 30% headroom. Qualification already used this exact batch.
    forecast=240+1.3*rollout_count*sum(physical)/len(physical)
    if forecast>remaining-20:raise TimeoutError('Calibrated follow-up exceeds the original remaining audit cap')
    root=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Require clean committed follow-up code')
    resources,observations=wait_for_idle_gpu(args.gpu,plan['gpu_uuid'],timeout=10)
    if resources['name']!=plan['gpu_model'] or float(resources['memory_free_mib'])<16384:
        raise ValueError('Follow-up GPU model/memory qualification failed')
    if shutil.disk_usage(parent).free/2**30<plan['minimum_disk_reserve_gib']+.5:
        raise ValueError('Follow-up disk reserve failed')
    output.mkdir(parents=True,exist_ok=False)
    phase_plan=output/'plan.json';write_json(phase_plan,design)
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    command=[sys.executable,'-m','experiments.ablation.fidelity','--probe-space','policy',
        '--checkpoint',native['args']['checkpoint'],'--bank',native['args']['bank'],
        '--sample-indices',*map(str,native['args']['sample_indices']),'--precision','fp32','--gradient-horizon','32',
        '--checkpoint-audit',str(phase_plan),'--action-std',str(native['args']['action_std']),
        '--gpu',str(args.gpu),'--gpu-uuid',plan['gpu_uuid'],'--max-seconds',str(remaining-20),'--output',str(output/'experiment')]
    record=dict(command=command,resources=resources,observations=observations,maximum_seconds=remaining)
    receipt=dict(status='running',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 parent_campaign_sha256=sha256(parent/'campaign.json'),parent_campaign=str(parent),
                 prior_audit_gpu_hours=consumed,cap_gpu_hours=plan['hard_gpu_hours'],
                 forecast_seconds=forecast,memory_forecast_mib=16384,commands=[record])
    def persist():
        receipt['allocated_gpu_hours']=record.get('seconds',0.)/3600
        receipt['combined_audit_gpu_hours']=consumed+receipt['allocated_gpu_hours']
        write_json(output/'campaign.json',receipt)
    persist()
    def terminate(signum,frame):
        raise KeyboardInterrupt(f'Calibrated follow-up received signal {signum}')
    previous=signal.signal(signal.SIGTERM,terminate)
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                       TORCHINDUCTOR_COMPILE_THREADS='4',ISAACSIM_ASSET_ROOT=str(root/'data/isaac-assets'))
    try:
        execute_probe(record,output/'experiment.log',gpu=args.gpu,cwd=root,environment=environment,
                      timeout=remaining,persist=persist)
        if json.loads((output/'experiment/probe.json').read_text())['status']!='complete':
            raise RuntimeError('Native calibrated follow-up incomplete')
        receipt['status']='complete'
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()
        signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
