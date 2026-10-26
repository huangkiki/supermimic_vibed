"""Resource-qualify, then execute the frozen independent-direction experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .common import sha256, write_json
from .run_fidelity_cohort import execute_probe
from .run_suite import wait_for_idle_gpu


def forecast_seconds(qualification, summary, costs, plan):
    """Extrapolate measured paired-gradient and paired-reference phases separately."""
    small, full = plan['qualification'], plan['experiment']
    gradient = math.fsum(float(c['bptt_seconds']) + float(c['ppo_seconds']) for c in costs)
    reference = summary['reference_phase_seconds']
    fixed = max(0., qualification['seconds'] - gradient - reference)
    gradient *= full['counts'][-1] / small['counts'][-1]
    reference *= full['directions'] * full['reference_count'] / (small['directions'] * small['reference_count'])
    return plan['forecast_safety_factor'] * (fixed + gradient + reference)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not 0 < plan['hard_gpu_hours'] <= 1 or plan['forecast_safety_factor'] < 1:
        raise ValueError('This bounded campaign permits at most one allocated GPU-hour')
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Native campaign requires a clean committed source tree')
    for name in ('checkpoint', 'bank'):
        if sha256(getattr(args, name)) != plan[name + '_sha256']:
            raise ValueError(f'Frozen {name} identity differs')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    affinity = sorted(os.sched_getaffinity(0))[-4:]
    os.sched_setaffinity(0, affinity)
    campaign = dict(status='running', source=source, plan=plan, plan_sha256=sha256(args.plan),
                    cpu_affinity=affinity, commands=[], allocated_gpu_hours=0.,
                    cost_scope='Allocated native subprocess time including initialization, qualification, failures and cleanup')

    def persist():
        campaign['allocated_gpu_hours'] = math.fsum(c.get('seconds', 0.) for c in campaign['commands']) / 3600
        write_json(output / 'campaign.json', campaign)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Campaign received signal {signum}')

    previous = signal.signal(signal.SIGTERM, terminate)
    persist()
    try:
        for phase in ('qualification', 'experiment'):
            design = plan[phase]
            remaining = 3600 * (plan['hard_gpu_hours'] - campaign['allocated_gpu_hours'])
            maximum = min(plan['qualification_max_seconds'], remaining) if phase == 'qualification' else remaining
            peak = plan['initial_peak_memory_forecast_mib']
            if phase == 'experiment':
                summary = json.loads((output / 'qualification/convergence-summary.json').read_text())
                with (output / 'qualification/estimator-costs.csv').open() as stream:
                    costs = list(csv.DictReader(stream))
                forecast = forecast_seconds(campaign['commands'][0], summary, costs, plan)
                peak = max(peak, campaign['commands'][0]['peak_device_memory_used_mib'] * 1.25)
                campaign.update(projected_experiment_seconds=forecast, projected_peak_memory_mib=peak)
                persist()
                if forecast > maximum - 20:
                    raise TimeoutError('Measured full-design forecast exceeds remaining campaign budget; no launch')
            resources, observations = wait_for_idle_gpu(args.gpu, plan['gpu_uuid'], timeout=10)
            free = float(resources['memory_free_mib'])
            if resources['name'] != plan['gpu_model'] or free < plan['minimum_free_memory_mib'] or peak > .85 * free:
                raise RuntimeError('GPU model or memory qualification failed')
            disk = shutil.disk_usage(output).free / 2**30
            if disk < plan['minimum_disk_reserve_gib'] + plan['artifact_forecast_gib']:
                raise RuntimeError('Disk reserve qualification failed')
            command = [sys.executable, '-m', 'experiments.ablation.fidelity', '--probe-space', 'policy',
                '--checkpoint', str(args.checkpoint.resolve()), '--bank', str(args.bank.resolve()),
                '--sample-indices', *map(str, plan['sample_indices']), '--gradient-horizon', str(plan['horizon']),
                '--precision', 'fp32', '--action-std', str(plan['action_std']),
                '--estimator-seeds', *map(str, range(design['gradient_start'], design['gradient_start'] + design['counts'][-1])),
                '--reference-seeds', *map(str, range(design['reference_start'], design['reference_start'] + design['reference_count'])),
                '--convergence-counts', *map(str, design['counts']), '--directions', str(design['directions']),
                '--direction-seed', str(design['direction_seed']), '--epsilons', *map(str, plan['epsilons']),
                '--repeats', str(plan['repeats']), '--gpu', str(args.gpu), '--gpu-uuid', plan['gpu_uuid'],
                '--max-seconds', str(maximum - 20), '--output', str(output / phase)]
            record = dict(phase=phase, resources=resources, observations=observations, free_disk_gib=disk,
                          maximum_seconds=maximum, command=command)
            campaign['commands'].append(record)
            persist()
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                               TORCHINDUCTOR_COMPILE_THREADS='4', ISAACSIM_ASSET_ROOT=str(root / 'data/isaac-assets'))
            execute_probe(record, output / f'{phase}.log', gpu=args.gpu, cwd=root,
                          environment=environment, timeout=maximum, persist=persist)
            if json.loads((output / phase / 'probe.json').read_text())['status'] != 'complete':
                raise RuntimeError('Native output did not complete')
        campaign['status'] = 'complete'
    except BaseException as exc:
        campaign.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
