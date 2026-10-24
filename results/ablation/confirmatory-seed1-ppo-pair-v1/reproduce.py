"""Reproduce two completed PPO methods, not a completed independent-seed cohort."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.ablation.analyze import collect_records, save_csv, write_analysis
from experiments.ablation.analyze_cohort import validate_clip_summary
from experiments.ablation.score_timing import score_timing, TIMING_SCOPE

METHODS = ('ppo_sim', 'ppo_wm_online')


def reproduce(raw, output):
    expected = json.loads(Path(__file__).with_name('input-sha256.json').read_text())
    for name, sha in expected.items():
        if hashlib.sha256((raw / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Published input identity differs: {name}')
    campaign = json.loads((raw / 'campaign.json').read_text())
    paths, records, clips, allocations, method_summary = [], [], [], [], {}
    expected_keys = {(i, r) for i in range(0, 4001, 1000) for r in range(3)}
    for method in METHODS:
        run = raw / f'{method}-seed1'
        manifest = json.loads((run / 'manifest.json').read_text())
        completion = json.loads((run / 'completion.json').read_text())
        cfg = manifest['config']
        if ((cfg['method'], cfg['seed'], cfg['iterations'], cfg['checkpoint_every']) != (method, 1, 4000, 1000)
                or completion['status'] != 'completed' or completion['iterations'] != 4000
                or manifest['gpu_uuid'] != campaign['resources']['uuid']
                or any(campaign['source_hashes'].get(k) != v for k, v in manifest['source_hashes'].items())):
            raise ValueError('Training scope, completion, source or physical GPU differs')
        with (run / 'train.csv').open() as stream:
            training = list(csv.DictReader(stream))
        if ([int(r['iteration']) for r in training] != list(range(1, 4001))
                or any(v and k not in ('method', 'kl_stopped') and not math.isfinite(float(v))
                       for row in training for k, v in row.items())):
            raise ValueError('Training iteration coverage is incomplete or nonfinite')
        with (raw / f'{run.name}-train.log').open() as stream:
            stdout = [json.loads(line) for line in stream if line.startswith('{"iteration":')]
        if len(stdout) != len(training):
            raise ValueError('Training stdout and CSV coverage differ')
        for csv_row, log_row in zip(training, stdout):
            for key, value in csv_row.items():
                if value == '':
                    equal = log_row[key] is None
                elif key in ('method', 'kl_stopped'):
                    equal = value == str(log_row[key])
                else:
                    equal = float(value) == float(log_row[key])
                if not equal:
                    raise ValueError(f'Training stdout and CSV disagree: {key}')
        method_paths = sorted(run.glob('ckpt_*/repeat_*/evaluation.json'))
        keys = []
        for path in method_paths:
            record = json.loads(path.read_text())
            keys.append((record['iteration'], record['evaluation_repeat_index']))
            if (record['method'] != method or record['seed'] != 1 or record['manifest'] != manifest
                    or record['evaluation_repeat_count'] != 3
                    or record['evaluation_manifest']['source_hashes'] != manifest['source_hashes']):
                raise ValueError('Evaluation identity differs from training')
            validate_clip_summary(path, record)
            records.append(record)
            with path.with_name('per_clip.csv').open() as stream:
                clips.extend(dict(method=method, iteration=record['iteration'],
                                  repeat=record['evaluation_repeat_index'], **row) for row in csv.DictReader(stream))
        if len(keys) != 15 or set(keys) != expected_keys:
            raise ValueError('Every checkpoint requires all three physical repeats')
        paths.extend(method_paths)
        for command in campaign['commands']:
            argv = command['command']
            if 'experiments.ablation.train' in argv and Path(argv[argv.index('--output') + 1]).name == run.name:
                stage, iteration, repeat = 'training', 4000, None
            elif 'experiments.ablation.evaluate' in argv:
                checkpoint = Path(argv[argv.index('--checkpoint') + 1])
                if checkpoint.parent.name != run.name:
                    continue
                stage, iteration, repeat = 'physical_evaluation', int(checkpoint.stem.split('_')[-1]), int(argv[argv.index('--repeat-index') + 1])
            else:
                continue
            if command.get('returncode') != 0 or not math.isfinite(command['seconds']) or command['seconds'] <= 0:
                raise ValueError('Selected method has an unpaid or failed allocation')
            allocations.append(dict(method=method, stage=stage, iteration=iteration, repeat=repeat,
                                    seconds=command['seconds'], gpu_hours=command['seconds'] / 3600,
                                    peak_device_memory_mib=command['peak_device_memory_used_mib']))
        selected = [r for r in allocations if r['method'] == method]
        keys = [(r['stage'], r['iteration'], r['repeat']) for r in selected]
        required = {('training', 4000, None), *(('physical_evaluation', i, r) for i, r in expected_keys)}
        if len(keys) != 16 or set(keys) != required:
            raise ValueError('Missing or duplicate selected method allocations')
        method_summary[method] = dict(training_iterations=4000,
            wm_only_iterations=cfg['wm_only_iters'] if method == 'ppo_wm_online' else 0,
            execution_commit=manifest['git_commit'], gpu_uuid=manifest['gpu_uuid'],
            final_failed_clips=[r['failed_clips'] for r in records if r['method'] == method and r['iteration'] == 4000])
    # Both calls enforce common data, source, physical GPU and evaluation protocol.
    rows = collect_records(paths, raw / 'campaign.json')
    timing = score_timing(paths, raw / 'campaign.json')
    for method in METHODS:
        endpoint = next(r for r in rows if r['method'] == method and r['iteration'] == 4000)
        cost = sum(r['seconds'] for r in allocations if r['method'] == method)
        if endpoint['cost_basis'] != 'complete_process_allocation' or not math.isclose(cost, endpoint['end_to_end_seconds'], abs_tol=1e-8):
            raise ValueError('Final method cost does not reconcile')
        method_summary[method]['final'] = endpoint
    if len(clips) != 540:
        raise ValueError('Incomplete per-motion evaluation coverage')
    plan = json.loads((Path(__file__).resolve().parents[3] / 'experiments/ablation/configs/confirmatory-locomotion-plan.json').read_text())
    failure_limit = plan['primary_endpoint']['max_tracking_failure_rate']
    thresholds = [plan['primary_endpoint']['return_threshold'], *plan['secondary_return_thresholds']]
    targets = [dict(method=method, return_threshold=threshold, max_failure_rate=failure_limit,
                    attained=any(r['physical_return'] >= threshold and r['tracking_failure_rate'] <= failure_limit
                                 for r in rows if r['method'] == method))
               for method in METHODS for threshold in thresholds]
    summary = dict(status='two_methods_recording_and_cost_complete', training_seeds=[1],
                   physical_evaluations=len(records), per_clip_rows=len(clips), methods=method_summary,
                   cohort_complete=False, uncertainty='Repeat SD is not independent-training-seed uncertainty',
                   native_tensor_qualification='Pending completion of the full six-condition campaign',
                   host_overlap='Early seed1 shared CPU/disk host with other work; measured costs preserved',
                   timing_scope=TIMING_SCOPE,
                   scope='Descriptive fixed-budget endpoints; no equivalence, causal attribution or quality-matched speedup claim')
    output.mkdir(parents=True, exist_ok=True)
    write_analysis(rows, records, output, 'PPO-Sim vs online PPO-WM: one training seed; retrospective checkpoint costs')
    save_csv(output / 'process-allocations.csv', allocations)
    save_csv(output / 'per-clip-evaluations.csv', clips)
    save_csv(output / 'score-timing.csv', timing)
    save_csv(output / 'registered-targets.csv', targets)
    (output / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for method, color in zip(METHODS, ('#4477AA', '#EE6677')):
        group = [r for r in timing if r['method'] == method]
        for ax, field, scale, label in zip(axes,
                ('score_available_allocation_seconds', 'score_available_gpu_hours'),
                (60., 1.), ('Method allocation at score availability (minutes)', 'Method GPU-hours at score availability')):
            ax.step([r[field] / scale for r in group], [r['physical_return'] for r in group],
                    where='post', marker='o', label=method, color=color)
            ax.set(xlabel=label, ylabel='Full-clip normalized physical return', ylim=(0, 1))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Offline scores: full training paid before evaluation; one seed, no seed-SD band')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'score-availability.{suffix}', dpi=180)
    plt.close(fig)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = reproduce(args.raw, args.output)
    print(json.dumps({k:v for k,v in summary.items() if k != 'methods'}, indent=2))
