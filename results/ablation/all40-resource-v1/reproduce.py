"""Rebuild the three completed all40 resource campaigns; never infer convergence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.ablation.analyze import collect_records, comparison_key, save_csv, write_analysis


def reproduce(root, output):
    if (root / 'exit.txt').read_text().strip() != '0':
        raise ValueError('Resource recipe did not complete successfully')
    physical, rows, allocations, measurements = [], [], [], []
    source_hashes = {}
    for method in ('ppo_sim', 'ppo_wm_online', 'supermimic'):
        folder = root / method
        campaign = json.loads((folder / 'campaign.json').read_text())
        validation = json.loads((folder / 'validation.json').read_text())
        if campaign['status'] != 'completed' or not all(
                check['training_and_full_physical_evaluation_valid'] for check in validation['checks']):
            raise ValueError(f'Incomplete native qualification: {method}')
        if validation['evaluation_processes'] != 2:
            raise ValueError('Each policy requires its initial and final physical evaluation')
        if not math.isclose(campaign['actual_gpu_hours'], sum(
                entry['seconds'] for entry in campaign['commands']) / 3600, rel_tol=1e-12):
            raise ValueError('Campaign allocation does not reconcile')
        for entry in campaign['commands']:
            if entry.get('returncode') != 0:
                raise ValueError('A resource process did not succeed')
            name = Path(entry['command'][entry['command'].index('--output') + 1]).name
            kind = 'train' if 'experiments.ablation.train' in entry['command'] else 'physical_evaluation'
            allocations.append(dict(campaign=method, stage=kind, run=name, seconds=entry['seconds'],
                                    gpu_hours=entry['seconds'] / 3600,
                                    peak_device_memory_mib=entry['peak_device_memory_used_mib']))
        paths = sorted(folder.glob('*/ckpt_*/evaluation.json'))
        records = [json.loads(path.read_text()) for path in paths]
        if len(records) != 2 or any(record['total_clips'] != 40 for record in records):
            raise ValueError('Incomplete all40 physical coverage')
        physical.extend(records)
        rows.extend(collect_records(paths, folder / 'campaign.json'))
        run = folder / f'{method}-seed0'
        manifest = json.loads((run / 'manifest.json').read_text())
        config = manifest['config']
        expected_iterations = 32 if method == 'ppo_sim' else 532
        if (config['num_envs'], config['horizon'], config['wm_window'], config['iterations']) != (4096, 32, 8, expected_iterations):
            raise ValueError('Unexpected resource qualification configuration')
        with (run / 'train.csv').open() as stream:
            training = list(csv.DictReader(stream))
        if [int(row['iteration']) for row in training] != list(range(1, expected_iterations + 1)):
            raise ValueError('Missing or duplicate training iteration')
        warm = training[-31:]
        expected_updates = config['ppo_epochs'] * config['minibatches'] if method.startswith('ppo') else 1
        if len(warm) != 31 or any(int(row['actor_updates']) != expected_updates for row in warm[1:]):
            raise ValueError('Warm resource intervals did not complete full Actor updates')
        phases = ('collection_seconds', 'wm_training_seconds', 'imagined_rollout_seconds',
                  'policy_optimization_seconds', 'train_wall_seconds')
        measured = dict(method=method, warm_intervals=30, actor_updates_per_interval=expected_updates)
        for phase in phases:
            differences = [float(b[phase]) - float(a[phase]) for a, b in zip(warm, warm[1:])]
            if any(not math.isfinite(value) or value < 0 for value in differences):
                raise ValueError('Invalid cumulative timing')
            measured[f'mean_{phase}'] = sum(differences) / len(differences)
            measured[f'max_{phase}'] = max(differences)
        for steps, seconds in (('simulator_steps', 'collection_seconds'), ('imagined_steps', 'imagined_rollout_seconds')):
            count = int(warm[-1][steps]) - int(warm[0][steps])
            elapsed = float(warm[-1][seconds]) - float(warm[0][seconds])
            measured[f'warm_{steps}_per_second'] = count / elapsed if elapsed else None
        measurements.append(measured)
        for path in (folder / 'campaign.json', folder / 'validation.json', run / 'manifest.json', run / 'train.csv', *paths):
            source_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len({comparison_key(record) for record in physical}) != 1:
        raise ValueError('Resource campaigns do not share data, physics, source and physical GPU')
    output.mkdir(parents=True, exist_ok=True)
    write_analysis(rows, physical, output, 'All 40 motions: resource qualification only; 32 policy iterations, seed 0')
    save_csv(output / 'process-allocations.csv', allocations)
    save_csv(output / 'warm-resources.csv', measurements)
    separate_pretrain = sum(row['gpu_hours'] for row in allocations if row['run'] == 'pretrain-seed0')
    total = sum(row['gpu_hours'] for row in allocations)
    endpoints = [row for row in rows if row['iteration'] > 0]
    if not math.isclose(sum(row['total_gpu_hours'] for row in endpoints) + separate_pretrain, total, rel_tol=1e-12):
        raise ValueError('Independent pretraining was lost or charged to PPO-Sim')
    summary = dict(status='native_resource_qualification_complete', motions=40, training_seed=0,
                   policy_updates=32, num_envs=4096, horizon=32, wm_window=8,
                   gpu=physical[0]['evaluation_gpu'], gpu_uuid=physical[0]['evaluation_gpu_uuid'],
                   actual_gpu_hours=total, independent_pretrain_gpu_hours=separate_pretrain,
                   process_count=len(allocations), physical_evaluation_count=len(physical),
                   per_clip_evaluation_rows=40 * len(physical), source_sha256=source_hashes,
                   final_failed_clips={record['method']: record['failed_clips'] for record in physical if record['iteration'] > 0},
                   scope='Resource qualification only; no efficiency or convergence ranking',
                   evaluation_timing_limit='Early-failure evaluation time does not estimate full-duration successful tracking',
                   overlap='Other development and confirmatory work overlapped on the same host; not isolated performance measurement')
    (output / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.raw, args.output), indent=2))
