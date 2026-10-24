"""Recompute planning estimates from archived CSV/JSON; no training or GPU calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.ablation.forecast import FROZEN_METHODS, calibrated_forecast


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main_root = args.raw_root / 'all40-resource-v1'
    frozen_root = args.raw_root / 'all40-frozen-resource-v1'
    runs = {m: main_root / m / f'{m}-seed0' for m in ('ppo_sim', 'ppo_wm_online', 'supermimic')}
    runs['pretrain'] = frozen_root / 'pretrain-seed0'
    frozen = {m: frozen_root / f'{m}-seed0' for m in FROZEN_METHODS}
    kwargs = dict(seeds=[1], gpu_uuid=args.gpu_uuid, iterations=4000, checkpoint_every=1000,
                  evaluation_repeats=3, evaluation_seconds=1500., disk_gib_per_seed=12.)
    forecasts = {'inferred': calibrated_forecast(runs, **kwargs),
                 'direct': calibrated_forecast(runs, frozen_runs=frozen, **kwargs)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, forecast in forecasts.items():
        # Relative provenance stays portable with the supplied raw archive root.
        forecast['sources'] = {m: str(Path(path).relative_to(args.raw_root))
                               for m, path in forecast['sources'].items()}
        (args.output / f'{name}.json').write_text(json.dumps(forecast, indent=2, allow_nan=False) + '\n')
    paths = {path for run in [*runs.values(), *frozen.values()]
             for path in (run / 'manifest.json', run / 'completion.json', run / 'train.csv', run.parent / 'campaign.json')}
    hashes = {str(p.relative_to(args.raw_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (args.output / 'input-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['basis', 'training_gpu_hours', 'evaluation_gpu_hours',
                                                    'total_with_25pct_reserve_gpu_hours'], lineterminator='\n')
        writer.writeheader()
        for basis, forecast in forecasts.items():
            writer.writerow(dict(basis=basis, training_gpu_hours=forecast['projected_training_gpu_hours'],
                                 evaluation_gpu_hours=forecast['projected_evaluation_gpu_hours'],
                                 total_with_25pct_reserve_gpu_hours=forecast['projected_campaign_gpu_hours']))
    print(json.dumps({name: forecast['projected_campaign_gpu_hours'] for name, forecast in forecasts.items()}))


if __name__ == '__main__':
    main()
