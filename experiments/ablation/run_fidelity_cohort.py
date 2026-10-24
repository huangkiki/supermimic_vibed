"""Run the registered three-seed gradient probes after the full physical cohort qualifies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .analyze_cohort import collect_cohort, read_json
from .analyze_fidelity_cohort import collect_fidelity_cohort, collect_seed
from .run_suite import gpu_resources, source_snapshot, wait_for_idle_gpu
from .validate_fidelity import validate


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_plan(training, plan):
    if any(training[k] != plan[k] for k in ('validation_seeds', 'development_seeds', 'subset')):
        raise ValueError('Training and gradient registrations differ')
    if (len(plan['validation_seeds']) < 3 or plan['method'] != 'supermimic'
            or plan['collector_iteration'] != training['iterations']
            or plan['gradient_horizons'] != [1, 8, 32] or plan['prediction_horizons'] != [1, 8, 16, 32]
            or plan['reward_objectives'] != ['full', 'state_only'] or plan['prefix_controller'] != 'policy'
            or plan['canonical_prediction_probe'] != 'gradient-components-full-h32'
            or plan['contact_level'] != 'body' or plan['primary_epsilon'] not in plan['epsilons']
            or plan['precision'] != 'fp32, TF32 disabled'
            or plan['reset_protocol'] != 'native physical state restore, nominal xi, Newton solver reset flags=0 before each rollout'):
        raise ValueError('Unsupported registered probe design')
    for key in ('samples', 'prefix_steps', 'directions'):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f'Positive integer probe setting required: {key}')
    if (type(plan['probe_seed']) is not int or plan['probe_seed'] < 0 or not plan['probe_gpu_uuid']
            or plan['wm_iterations'] != sorted(set(plan['wm_iterations'])) or len(plan['wm_iterations']) < 2
            or any(type(i) is not int or i < 0 for i in plan['wm_iterations'])
            or plan['epsilons'] != sorted(set(plan['epsilons'])) or len(plan['epsilons']) < 2
            or any(not math.isfinite(e) or e <= 0 for e in plan['epsilons'])):
        raise ValueError('Invalid probe stages, seed or perturbations')
    for key in ('probe_gpu_hours_cap_per_seed', 'probe_gpu_hours_cap_total', 'projected_gpu_hours_per_seed',
                'projected_peak_memory_mib', 'minimum_free_memory_mib', 'projected_disk_gib_per_seed',
                'minimum_disk_reserve_gib'):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f'Finite positive probe resource allowance required: {key}')
    if (plan['projected_gpu_hours_per_seed'] > plan['probe_gpu_hours_cap_per_seed']
            or len(plan['validation_seeds']) * plan['projected_gpu_hours_per_seed'] > plan['probe_gpu_hours_cap_total']):
        raise ValueError('Measured projection exceeds the registered probe budget')


def commands_for_seed(run, output, plan, python):
    """Create one bank with the first probe; reuse that exact file in the next five."""
    commands, bank = [], None
    for horizon in plan['gradient_horizons']:
        for objective in plan['reward_objectives']:
            dest = output / f'gradient-components-{objective}-h{horizon}'
            command = [str(python), '-m', 'experiments.ablation.fidelity', '--checkpoints',
                       *[str(run / f'ckpt_{iteration:06d}.pt') for iteration in plan['wm_iterations']],
                       '--collector', str(run / f"ckpt_{plan['collector_iteration']:06d}.pt"), '--output', str(dest),
                       '--samples', str(plan['samples']), '--prefix-controller', plan['prefix_controller'],
                       '--prefix-steps', str(plan['prefix_steps']), '--directions', str(plan['directions']),
                       '--epsilons', *map(str, plan['epsilons']), '--probe-seed', str(plan['probe_seed']),
                       '--gradient-horizon', str(horizon), '--reward-objective', objective,
                       '--contact-level', plan['contact_level']]
            if bank is not None:
                command += ['--bank', str(bank)]
            commands.append((dest, command))
            if bank is None:
                bank = dest / 'heldout-bank.pt'
    return commands


def preflight(gpu, plan, model, output, remaining_seeds, remaining_seconds):
    resources, observations = wait_for_idle_gpu(gpu, plan['probe_gpu_uuid'], min(30., remaining_seconds))
    free = float(resources['memory_free_mib'])
    if (resources['name'] != model or not math.isfinite(free) or free < plan['minimum_free_memory_mib']
            or plan['projected_peak_memory_mib'] > free * .85):
        raise ValueError('Probe GPU model or memory qualification failed')
    needed = plan['minimum_disk_reserve_gib'] + remaining_seeds * plan['projected_disk_gib_per_seed']
    if shutil.disk_usage(output).free / 2**30 < needed:
        raise ValueError('Probe disk qualification failed')
    return resources, observations


def execute_probe(record, log, *, gpu, cwd, environment, timeout, persist):
    """Record all allocated process time, including failures and termination cleanup."""
    if timeout <= 10:
        raise TimeoutError('Insufficient probe budget including termination reserve')
    process = None
    started = time.monotonic()
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(record['command'], cwd=cwd, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, peak_device_memory_used_mib=0.)
            persist()
            while True:
                sample = gpu_resources(gpu)
                if sample['uuid'] != record['resources']['uuid']:
                    raise RuntimeError('Probe GPU identity changed')
                used = float(sample['memory_total_mib']) - float(sample['memory_free_mib'])
                record['peak_device_memory_used_mib'] = max(record['peak_device_memory_used_mib'], used)
                remaining = timeout - 10 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Probe allocation deadline reached')
                try:
                    record['returncode'] = process.wait(timeout=min(1., remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if record['returncode']:
                raise RuntimeError(f'Probe failed; preserved {log}')
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        finally:
            record['returncode'] = process.returncode if process is not None else None
            record['seconds'] = time.monotonic() - started
            persist()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--probe-plan', type=Path, required=True)
    parser.add_argument('--training-campaigns', nargs='+', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate completed cohort and checkpoint bytes; no GPU queries or launches')
    args = parser.parse_args()
    wall_start = time.monotonic()
    training, plan = read_json(args.training_plan), read_json(args.probe_plan)
    validate_plan(training, plan)
    physical, raw, receipts = collect_cohort(args.training_campaigns, training)
    root = Path(__file__).resolve().parents[2]
    sources = source_snapshot(root)
    inputs = {str(p.resolve()): digest(p) for p in (args.training_plan, args.probe_plan)}
    runs, manifests = {}, {}
    for receipt in receipts:
        inputs[receipt['campaign']] = receipt['sha256']
        for seed in receipt['seeds']:
            run = Path(receipt['campaign']).parent / f"{plan['method']}-seed{seed}"
            manifests[seed] = read_json(run / 'manifest.json')
            core = {k: v for k, v in manifests[seed]['source_hashes'].items() if k.startswith('supermimic/')}
            if not core or any(sources.get(k) != v for k, v in core.items()):
                raise ValueError('Probe implementation changed trained physical core')
            stages = {r['iteration']: r['checkpoint_sha256'] for r in physical
                      if r['seed'] == seed and r['method'] == plan['method']}
            for iteration in {*plan['wm_iterations'], plan['collector_iteration']}:
                checkpoint = run / f'ckpt_{iteration:06d}.pt'
                if digest(checkpoint) != stages.get(iteration):
                    raise ValueError('Probe checkpoint differs from physical evaluation bytes')
                inputs[str(checkpoint)] = stages[iteration]
            runs[seed] = run
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh probe output; no automatic resume or budget reset')
    commands = {seed: commands_for_seed(runs[seed], output / f'seed{seed}', plan, sys.executable)
                for seed in plan['validation_seeds']}
    if args.dry_run:
        print(json.dumps(dict(status='validated_dry_run_no_gpu', commands={s: [c for _, c in batch]
                         for s, batch in commands.items()}, input_sha256=inputs), indent=2))
        return
    output.mkdir(parents=True)
    suite = dict(status='running', plan=plan, input_sha256=inputs, source_hashes=sources,
                 training_receipts=receipts, seeds={}, actual_gpu_hours=0.,
                 budget_scope='Allocated fidelity subprocess time including setup and cleanup; CPU-only validation is included in suite wall time')

    def persist():
        for seed, campaign in suite['seeds'].items():
            campaign['actual_gpu_hours'] = math.fsum(c.get('seconds', 0.) for c in campaign['commands']) / 3600
            (output / f'seed{seed}/campaign.json').write_text(json.dumps(campaign, indent=2, allow_nan=False) + '\n')
        suite['actual_gpu_hours'] = math.fsum(c['actual_gpu_hours'] for c in suite['seeds'].values())
        suite['wall_seconds'] = time.monotonic() - wall_start
        (output / 'suite.json').write_text(json.dumps(suite, indent=2, allow_nan=False) + '\n')

    persist()
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Probe suite received signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    active_campaign = None
    try:
        for index, seed in enumerate(plan['validation_seeds']):
            dest = output / f'seed{seed}'
            dest.mkdir()
            campaign = dict(status='running', plan={**plan, 'training_seed': seed,
                            'hard_gpu_hours': plan['probe_gpu_hours_cap_per_seed']}, source_hashes=sources,
                            commands=[], actual_gpu_hours=0.)
            suite['seeds'][seed] = campaign
            active_campaign = campaign
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(root),
                               OMP_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='4', PYTHONUNBUFFERED='1',
                               WARP_CACHE_PATH=str(dest / 'cache/warp'), TORCHINDUCTOR_CACHE_DIR=str(dest / 'cache/inductor'))
            for probe, command in commands[seed]:
                if source_snapshot(root) != sources or any(digest(p) != sha for p, sha in inputs.items()):
                    raise RuntimeError('Probe source, registration, training receipt or checkpoint changed')
                remaining = min(plan['probe_gpu_hours_cap_per_seed'] - campaign['actual_gpu_hours'],
                                plan['probe_gpu_hours_cap_total'] - suite['actual_gpu_hours']) * 3600
                if remaining <= 10:
                    raise TimeoutError('Registered probe budget exhausted')
                resources, observations = preflight(args.gpu, plan, manifests[seed]['gpu'], output,
                                                     len(plan['validation_seeds']) - index, remaining)
                campaign.setdefault('resources', resources)
                record = dict(command=command, resources=resources, preflight_observations=observations)
                campaign['commands'].append(record)
                persist()
                execute_probe(record, dest / f'{probe.name}.log', gpu=args.gpu, cwd=root,
                              environment=environment, timeout=min(600., remaining), persist=persist)
                if (campaign['actual_gpu_hours'] > plan['probe_gpu_hours_cap_per_seed']
                        or suite['actual_gpu_hours'] > plan['probe_gpu_hours_cap_total']):
                    raise TimeoutError('Probe cleanup exceeded registered budget; no further launch')
                if validate(probe)['bank_identity'] != 'sha256_verified':
                    raise ValueError('Native shared bank file was not verified')
                bank = commands[seed][0][0] / 'heldout-bank.pt'
                bank_sha = read_json(probe / 'probe.json')['bank_sha256']
                if digest(bank) != bank_sha or (str(bank) in inputs and inputs[str(bank)] != bank_sha):
                    raise ValueError('Probe changed its shared state bank')
                inputs[str(bank)] = bank_sha
            if source_snapshot(root) != sources or any(digest(p) != sha for p, sha in inputs.items()):
                raise RuntimeError('Probe source or inputs changed before final seed validation')
            campaign['status'] = 'completed'
            persist()
            collect_seed(dest, physical, manifests[seed], plan)
            active_campaign = None
        collect_fidelity_cohort([output / f'seed{s}' for s in plan['validation_seeds']], physical, raw, plan)
        suite['status'] = 'completed'
    except BaseException as error:
        suite.update(status='stopped', error=str(error))
        if active_campaign is not None:
            active_campaign.update(status='stopped', error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        persist()


if __name__ == '__main__':
    main()
