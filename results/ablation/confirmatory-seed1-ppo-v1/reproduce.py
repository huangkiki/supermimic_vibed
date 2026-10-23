"""Reproduce one completed PPO-Sim method, without claiming a completed cohort."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.ablation.analyze import collect_records, save_csv, write_analysis
from experiments.ablation.analyze_cohort import validate_clip_summary


def reproduce(raw, output):
    expected = json.loads(Path(__file__).with_name('input-sha256.json').read_text())
    for relative, digest in expected.items():
        if hashlib.sha256((raw / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Published input identity differs: {relative}')
    run = raw / 'ppo_sim-seed1'
    manifest = json.loads((run / 'manifest.json').read_text())
    completion = json.loads((run / 'completion.json').read_text())
    campaign = json.loads((raw / 'campaign.json').read_text())
    config = manifest['config']
    if ((config['method'], config['seed'], config['iterations'], config['checkpoint_every'])
            != ('ppo_sim', 1, 4000, 1000) or completion['status'] != 'completed'
            or completion['iterations'] != 4000):
        raise ValueError('PPO-Sim training scope or completion differs')
    with (run / 'train.csv').open() as stream:
        training = list(csv.DictReader(stream))
    if [int(row['iteration']) for row in training] != list(range(1, 4001)):
        raise ValueError('Incomplete training iteration coverage')
    if any(value and name not in ('method', 'kl_stopped') and not math.isfinite(float(value))
           for row in training for name, value in row.items()):
        raise ValueError('Nonfinite training measurement')
    paths = sorted(run.glob('ckpt_*/repeat_*/evaluation.json'))
    expected_keys = {(iteration, repeat) for iteration in range(0, 4001, 1000) for repeat in range(3)}
    records, clips, keys = [], [], []
    for path in paths:
        record = json.loads(path.read_text())
        key = record['iteration'], record['evaluation_repeat_index']
        keys.append(key)
        if (record['method'] != 'ppo_sim' or record['seed'] != 1
                or record['manifest'] != manifest or record['evaluation_repeat_count'] != 3
                or record['evaluation_manifest']['source_hashes'] != manifest['source_hashes']):
            raise ValueError('Evaluation identity differs from the completed training run')
        validate_clip_summary(path, record)
        records.append(record)
        with path.with_name('per_clip.csv').open() as stream:
            clips.extend(dict(iteration=key[0], repeat=key[1], **row) for row in csv.DictReader(stream))
    if len(keys) != 15 or set(keys) != expected_keys:
        raise ValueError('All five checkpoints and their three repeats are required')
    rows = collect_records(paths, raw / 'campaign.json')
    allocations = []
    for command in campaign['commands']:
        args = command['command']
        destination = Path(args[args.index('--output') + 1])
        if 'experiments.ablation.train' in args and destination.name == run.name:
            stage, iteration, repeat = 'training', 4000, None
        elif 'experiments.ablation.evaluate' in args:
            checkpoint = Path(args[args.index('--checkpoint') + 1])
            if checkpoint.parent.name != run.name:
                continue
            stage, iteration = 'physical_evaluation', int(checkpoint.stem.split('_')[-1])
            repeat = int(args[args.index('--repeat-index') + 1])
        else:
            continue
        if command.get('returncode') != 0 or not math.isfinite(command['seconds']) or command['seconds'] <= 0:
            raise ValueError('Selected method has an incomplete process allocation')
        allocations.append(dict(stage=stage, iteration=iteration, repeat=repeat,
            seconds=command['seconds'], gpu_hours=command['seconds'] / 3600,
            peak_device_memory_mib=command['peak_device_memory_used_mib']))
    allocation_keys = [(row['stage'], row['iteration'], row['repeat']) for row in allocations]
    required = {('training', 4000, None), *(('physical_evaluation', i, r) for i, r in expected_keys)}
    if len(allocation_keys) != 16 or set(allocation_keys) != required:
        raise ValueError('Missing or duplicate method process allocations')
    endpoint = rows[-1]
    if (endpoint['cost_basis'] != 'complete_process_allocation'
            or not math.isclose(sum(row['gpu_hours'] for row in allocations), endpoint['total_gpu_hours'], abs_tol=1e-12)
            or any(row['wm_training_seconds'] or row['imagined_steps'] for row in rows)):
        raise ValueError('PPO-Sim standalone cost does not reconcile')

    final_clips = {}
    for clip in clips:
        if clip['iteration'] == 4000:
            final_clips.setdefault(clip['motion'], {})[clip['repeat']] = clip
    if len(final_clips) != 18 or any(set(group) != {0, 1, 2} for group in final_clips.values()):
        raise ValueError('Final repeat diagnostics require all 18 motions and all three repeats')
    motion_rows = []
    for motion, group in sorted(final_clips.items()):
        values = [float(group[index]['normalized_return']) for index in range(3)]
        failures = [group[index]['failed'] == 'True' for index in range(3)]
        row = dict(motion=motion, reference_steps=int(group[0]['reference_steps']),
                   repeat_return_mean=statistics.mean(values), repeat_return_sd=statistics.stdev(values),
                   failed_repeats=sum(failures), failure_varies=len(set(failures)) > 1)
        for index in range(3):
            row.update({f'return_repeat_{index}': values[index], f'failed_repeat_{index}': failures[index],
                        f'steps_repeat_{index}': int(group[index]['steps'])})
        motion_rows.append(row)
    varying = {row['motion'] for row in motion_rows if row['failure_varies']}
    final_records = {record['evaluation_repeat_index']: record for record in records if record['iteration'] == 4000}
    differences, pairs = [], []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        contributions = {motion: (float(group[right]['normalized_return']) - float(group[left]['normalized_return'])) / 18
                         for motion, group in final_clips.items()}
        recorded_difference = final_records[right]['physical_return'] - final_records[left]['physical_return']
        reconstructed = sum(contributions.values())
        if not math.isclose(recorded_difference, reconstructed, rel_tol=0, abs_tol=1e-7):
            raise ValueError('Per-motion differences do not reconcile with recorded physical returns')
        differences.extend(dict(from_repeat=left, to_repeat=right, motion=motion,
                                contribution_to_mean_difference=value, failure_varies=motion in varying)
                           for motion, value in sorted(contributions.items()))
        pairs.append(dict(from_repeat=left, to_repeat=right, recorded_return_difference=recorded_difference,
                          reconstructed_return_difference=reconstructed,
                          failure_varying_motion_contribution=sum(contributions[motion] for motion in varying),
                          other_motion_contribution=sum(value for motion, value in contributions.items() if motion not in varying),
                          float_reduction_residual=recorded_difference - reconstructed))
    output.mkdir(parents=True, exist_ok=True)
    write_analysis(rows, records, output, 'PPO-Sim: one confirmatory training seed, three physical repeats per checkpoint')
    save_csv(output / 'per-clip-evaluations.csv', clips)
    save_csv(output / 'process-allocations.csv', allocations)
    save_csv(output / 'final-per-motion-repeatability.csv', motion_rows)
    save_csv(output / 'final-repeat-differences.csv', differences)
    (output / 'repeat-diagnostics.json').write_text(json.dumps(dict(
        iteration=4000, training_seeds=[1], evaluation_repeats=3, motions=18,
        failure_varying_motions=sorted(varying),
        always_failed_motions=[row['motion'] for row in motion_rows if row['failed_repeats'] == 3],
        always_completed_motions=[row['motion'] for row in motion_rows if row['failed_repeats'] == 0],
        pairs=pairs, reconciliation_absolute_tolerance=1e-7,
        scope='Descriptive arithmetic decomposition of one checkpoint; all three pairs retained; not a causal explanation or independent-training-seed uncertainty'
    ), indent=2) + '\n')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, field, deviation, label in zip(axes,
            ('physical_return', 'tracking_failure_rate'),
            ('evaluation_return_std', 'evaluation_failure_rate_std'),
            ('Full-clip normalized physical return', 'Tracking failure rate')):
        ax.scatter([record['iteration'] for record in records], [record[field] for record in records],
                   color='#EE6677', s=24, alpha=.7, label='Individual physical repeats', zorder=3)
        ax.errorbar([row['iteration'] for row in rows], [row[field] for row in rows],
                    yerr=[row[deviation] for row in rows], fmt='o-', capsize=4,
                    color='#4477AA', label='Repeat mean and sample SD', zorder=2)
        ax.set(xlabel='Training iteration', ylabel=label, ylim=(-.03, 1.04), xticks=range(0, 4001, 1000))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('PPO-Sim seed 1: within-checkpoint repeat variability, not training-seed uncertainty')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'evaluation-repeatability.{suffix}', dpi=180)
    plt.close(fig)
    summary = dict(status='one_method_training_and_physical_records_complete', training_seeds=[1],
        checkpoints=5, physical_evaluations=15, per_clip_rows=len(clips), iterations=4000,
        execution_commit=manifest['git_commit'], gpu_uuid=manifest['gpu_uuid'],
        method_gpu_hours=endpoint['total_gpu_hours'], training_seconds=endpoint['train_wall_seconds'],
        evaluation_seconds=endpoint['cumulative_evaluation_seconds'], final=endpoint,
        final_failed_clips=[record['failed_clips'] for record in records if record['iteration'] == 4000],
        input_sha256=expected, cohort_complete=False,
        scope='Recording and cost qualification for one method; complete-cohort native tensor qualification remains pending',
        uncertainty='Three physical repeats of one training seed; no independent-training-seed SD or algorithm ranking',
        timing='Intermediate points reconstruct checkpoint timing plus preceding evaluation allocations; not online score availability',
        host_overlap='Seed1 shared the host with earlier resource probes and archival; costs preserved without adjustment',
        repeat_variation='All repeats retained; matching recorded identity does not establish the numerical cause of variation')
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
