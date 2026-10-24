"""Distinguish reconstructed checkpoint cost from offline score availability."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.ablation.analyze import collect_records, group_repeats, save_csv


TIMING_SCOPE = (
    "Score availability charges the complete training process and all preceding physical "
    "evaluation processes for that method, including the current repeat group. Frozen "
    "methods also pay their complete shared pretraining allocation once. These are sums "
    "of measured subprocess wall times, not absolute timestamps; unrelated methods, queue "
    "waiting and orchestration gaps outside those processes are excluded. Checkpoint "
    "reconstruction uses the earlier checkpoint training snapshot instead."
)


def score_timing(paths, campaign_path):
    """Return both time axes for complete repeat groups in an executed prefix.

    Earlier evaluation records cannot be omitted: otherwise their cost would disappear
    from the reconstructed curve. The supported schedule trains a method to completion
    before evaluating its checkpoints in increasing iteration order, as run_suite does.
    """
    paths = [Path(path) for path in paths]
    records = [json.loads(path.read_text()) for path in paths]
    if any(not record.get('evaluation_gpu_uuid') or not record['manifest'].get('gpu_uuid')
           for record in records):
        raise ValueError('Physical GPU identity is required for score timing')
    groups = group_repeats(records)
    campaign = json.loads(Path(campaign_path).read_text())
    training, evaluations = {}, {}
    for position, command in enumerate(campaign['commands']):
        argv = command['command']
        if 'experiments.ablation.train' in argv:
            run = Path(argv[argv.index('--output') + 1]).name
            if run in training:
                raise ValueError(f'Duplicate training allocation: {run}')
            training[run] = position, command
        elif 'experiments.ablation.evaluate' in argv:
            checkpoint = Path(argv[argv.index('--checkpoint') + 1])
            iteration = int(checkpoint.stem.split('_')[-1])
            repeat = int(argv[argv.index('--repeat-index') + 1]) if '--repeat-index' in argv else 0
            key = checkpoint.parent.name, iteration, repeat
            if key in evaluations:
                raise ValueError(f'Duplicate evaluation allocation: {key}')
            evaluations[key] = position, command

    def paid_seconds(command):
        seconds = command.get('seconds')
        if (command.get('returncode') != 0 or seconds is None
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('Score timing requires completed successful process allocations')
        return seconds

    available = {}
    for method, seed in {(method, seed) for method, seed, _ in groups}:
        run = f'{method}-seed{seed}'
        if run not in training:
            raise ValueError(f'Missing completed training allocation: {run}')
        train_position, train_command = training[run]
        training_seconds = paid_seconds(train_command)
        if '_frozen' in method:
            pretrain = f'pretrain-seed{seed}'
            if pretrain not in training or training[pretrain][0] >= train_position:
                raise ValueError('Frozen pretraining must precede policy training')
            training_seconds += paid_seconds(training[pretrain][1])
        selected = {
            (run, iteration, record.get('evaluation_repeat_index', 0))
            for (m, s, iteration), members in groups.items() if (m, s) == (method, seed)
            for record in members
        }
        if not selected.issubset(evaluations):
            raise ValueError('An evaluation record has no matching process allocation')
        last_position = max(evaluations[key][0] for key in selected)
        prefix = sorted((position, key, command)
                        for key, (position, command) in evaluations.items()
                        if key[0] == run and position <= last_position)
        if {key for _, key, _ in prefix} != selected:
            raise ValueError('Earlier physical evaluation records are missing from the prefix')
        if any(position <= train_position for position, _, _ in prefix):
            raise ValueError('Offline evaluation must follow completed training')
        if [key[1:] for _, key, _ in prefix] != sorted(key[1:] for _, key, _ in prefix):
            raise ValueError('Checkpoint and repeat allocations are out of schedule order')
        elapsed = training_seconds
        for _, key, command in prefix:
            elapsed += paid_seconds(command)
            available[method, seed, key[1]] = elapsed

    rows = []
    for record in collect_records(paths, campaign_path):
        key = record['method'], record['seed'], record['iteration']
        elapsed = available[key]
        reconstructed = record['end_to_end_seconds']
        if elapsed + 1e-6 < reconstructed:
            raise ValueError('Checkpoint cost exceeds the complete training allocation')
        rows.append(dict(
            method=record['method'], seed=record['seed'], iteration=record['iteration'],
            physical_return=record['physical_return'],
            tracking_failure_rate=record['tracking_failure_rate'],
            evaluation_repeats=record['evaluation_repeats'],
            evaluation_return_std=record['evaluation_return_std'],
            reconstructed_checkpoint_seconds=reconstructed,
            score_available_allocation_seconds=elapsed,
            score_available_gpu_hours=elapsed / 3600,
            checkpoint_sha256=record['checkpoint_sha256'],
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = score_timing(args.evaluations, args.campaign)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / 'score-timing.csv', rows)
    (args.output / 'timing-scope.json').write_text(json.dumps(dict(
        scope=TIMING_SCOPE, checkpoints=len(rows),
        recorded_order='all policy training, then physical evaluation checkpoints and repeats',
        interpretation='The availability axis measures the offline evaluation pipeline, not learning speed alone',
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
