"""Qualify and execute the fixed-policy checkpoint audit with cumulative costs."""
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


def forecast_seconds(record, timings, small, full, margin):
    g = len(full['gradient_seeds']) / len(small['gradient_seeds'])
    r = len(full['reference_seeds']) / len(small['reference_seeds'])
    d = full['directions'] / small['directions']
    measured = math.fsum(float(row['seconds']) for row in timings)
    # Includes model loads, Python/CSV overhead and moment accumulation. Scaling
    # all overhead by the largest sample ratio deliberately overestimates setup.
    estimate = max(0., record['seconds'] - measured) * max(g, r, d)
    for row in timings:
        operation = row['operation']
        multiplier = (g if operation == 'ppo_sim' or operation.endswith('_paired_gradients')
                      else r*d if operation.endswith('_wm_reference') or operation == 'physical_independent_direction'
                      else 1 if operation == 'physical_repeat' else r)
        estimate += float(row['seconds']) * multiplier
    return margin * estimate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--checkpoint-directory', required=True, type=Path)
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not 0 < plan['hard_gpu_hours'] <= 1 or plan['forecast_safety_factor'] < 1:
        raise ValueError('Require a bounded one-GPU-hour audit with forecast margin')
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Require a clean committed native source tree')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    checkpoint = args.checkpoint_directory / 'ckpt_004000.pt'
    if sha256(checkpoint) != plan['checkpoint_sha256'] or sha256(args.bank) != plan['bank_sha256']:
        raise ValueError('Frozen checkpoint/bank identity differs')
    paths = [str((args.checkpoint_directory / f'ckpt_{i:06d}.pt').resolve()) for i in plan['wm_iterations']]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    designs = {phase: dict(**plan[phase], epsilons=plan['epsilons'], primary_epsilon=plan['primary_epsilon'],
                direction_seed=plan['direction_seed'], repeats=plan['repeats'], replicas=plan['replicas'],
                wm_checkpoints=paths) for phase in ('qualification', 'experiment')}
    campaign = dict(status='running', source=source, plan=plan, plan_sha256=sha256(args.plan),
                    wm_checkpoint_hashes={p: sha256(p) for p in paths}, commands=[], allocated_gpu_hours=0.,
                    prior_diagnostic_gpu_hours=plan['prior_diagnostic_gpu_hours'],
                    cost_scope='allocated process seconds incl qualification/failures/cleanup; no new training')

    def persist():
        campaign['allocated_gpu_hours'] = math.fsum(c.get('seconds', 0.) for c in campaign['commands']) / 3600
        campaign['diagnostic_gpu_hours_including_prior'] = campaign['allocated_gpu_hours'] + sum(plan['prior_diagnostic_gpu_hours'].values())
        write_json(output / 'campaign.json', campaign)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Audit received signal {signum}')

    previous = signal.signal(signal.SIGTERM, terminate)
    persist()
    try:
        for phase in ('qualification', 'experiment'):
            maximum = 3600 * (plan['hard_gpu_hours'] - campaign['allocated_gpu_hours'])
            peak = plan['initial_peak_memory_forecast_mib']
            if phase == 'qualification':
                maximum = min(maximum, plan['qualification_max_seconds'])
            else:
                with (output / 'qualification/timings.csv').open() as stream:
                    timings = list(csv.DictReader(stream))
                forecast = forecast_seconds(campaign['commands'][0], timings, designs['qualification'],
                                            designs['experiment'], plan['forecast_safety_factor'])
                peak = max(peak, 1.3 * campaign['commands'][0]['peak_device_memory_used_mib'])
                campaign.update(projected_experiment_seconds=forecast, projected_peak_memory_mib=peak)
                persist()
                if forecast > maximum - 20:
                    raise TimeoutError('Measured full audit forecast exceeds remaining budget; no launch')
            resources, observations = wait_for_idle_gpu(args.gpu, plan['gpu_uuid'], timeout=10)
            if resources['name'] != plan['gpu_model'] or peak > .85 * float(resources['memory_free_mib']):
                raise RuntimeError('GPU identity/memory qualification failed')
            if shutil.disk_usage(output).free / 2**30 < plan['minimum_disk_reserve_gib'] + plan['artifact_forecast_gib']:
                raise RuntimeError('Disk reserve qualification failed')
            phase_plan = output / (phase + '-plan.json')
            write_json(phase_plan, designs[phase])
            command = [sys.executable, '-m', 'experiments.ablation.fidelity', '--probe-space', 'policy',
                '--checkpoint', str(checkpoint.resolve()), '--bank', str(args.bank.resolve()),
                '--sample-indices', *map(str, plan['sample_indices']), '--gradient-horizon', '32',
                '--precision', 'fp32', '--action-std', str(plan['action_std']),
                '--checkpoint-audit', str(phase_plan), '--gpu', str(args.gpu), '--gpu-uuid', plan['gpu_uuid'],
                '--max-seconds', str(maximum - 20), '--output', str(output / phase)]
            record = dict(phase=phase, resources=resources, observations=observations, command=command,
                          maximum_seconds=maximum, memory_forecast_mib=peak)
            campaign['commands'].append(record)
            persist()
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                               TORCHINDUCTOR_COMPILE_THREADS='4', ISAACSIM_ASSET_ROOT=str(root / 'data/isaac-assets'))
            execute_probe(record, output / (phase + '.log'), gpu=args.gpu, cwd=root,
                          environment=environment, timeout=maximum, persist=persist)
            if json.loads((output / phase / 'probe.json').read_text())['status'] != 'complete':
                raise RuntimeError('Native audit incomplete')
        campaign['status'] = 'complete'
    except BaseException as exc:
        campaign.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
