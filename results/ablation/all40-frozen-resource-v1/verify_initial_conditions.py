"""Verify the frozen comparison's actual tensors on CPU, without rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


METHODS = ('ppo_wm_frozen_hard', 'ppo_wm_frozen', 'bptt_wm_frozen')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require_equal(reference, actual, label):
    if reference.keys() != actual.keys():
        raise ValueError(f'{label}: state keys differ')
    mismatches = [name for name, expected in reference.items()
                  if expected.dtype != actual[name].dtype or expected.shape != actual[name].shape
                  or not torch.equal(expected, actual[name])]
    if mismatches:
        raise ValueError(f'{label}: tensor mismatch: {mismatches}')


def verify(root, output):
    started = time.perf_counter()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('Run this verification in a fresh CPU-only process')
    campaign = json.loads((root / 'campaign.json').read_text())
    validation = json.loads((root / 'validation.json').read_text())
    if (campaign['status'] != 'completed' or campaign['args']['seeds'] != [0]
            or set(campaign['args']['methods']) != {'pretrain', *METHODS}
            or len(validation['checks']) != 4 or validation['evaluation_processes'] != 6
            or not all(check['training_and_full_physical_evaluation_valid'] for check in validation['checks'])):
        raise ValueError('Complete native four-run/six-evaluation qualification is required')
    bootstrap_path = root / 'pretrain-seed0/bootstrap.pt'
    bootstrap_sha = digest(bootstrap_path)
    bootstrap = torch.load(bootstrap_path, map_location='cpu', weights_only=False, mmap=True)
    if bootstrap['config']['method'] != 'pretrain' or bootstrap['iteration'] != 500:
        raise ValueError('Expected the declared independent 500-iteration bootstrap')
    # The file hash identifies the whole replay bank; parameter checks need no replay copy.
    replay_size, replay_ptr = bootstrap['replay']['size'], bootstrap['replay']['ptr']
    del bootstrap['replay']
    checks = []
    for method in METHODS:
        run = root / f'{method}-seed0'
        first_path, final_path = run / 'ckpt_000000.pt', run / 'ckpt_000032.pt'
        first = torch.load(first_path, map_location='cpu', weights_only=False, mmap=True)
        final = torch.load(final_path, map_location='cpu', weights_only=False, mmap=True)
        if first['iteration'] != 0 or final['iteration'] != 32:
            raise ValueError('Checkpoint iteration differs from the qualified scope')
        for checkpoint in (first, final):
            if checkpoint['manifest']['bootstrap_sha256'] != bootstrap_sha:
                raise ValueError('Frozen methods did not load the same bootstrap bytes')
        counts = {}
        for module in ('pi', 'critic', 'wm'):
            require_equal(bootstrap[module], first[module], f'{method}: initial {module}')
            counts[module] = len(first[module])
        require_equal(bootstrap['wm'], final['wm'], f'{method}: final WM and normalization')
        if not torch.equal(first['log_std'], bootstrap['log_std']):
            raise ValueError('Initial Gaussian standard deviation differs')
        for field in ('simulator_steps', 'collection_seconds', 'wm_training_seconds'):
            if first['ledger'][field] != final['ledger'][field]:
                raise ValueError(f'Frozen method acquired new physical data or model training cost: {field}')
        checks.append(dict(method=method, first_sha256=digest(first_path), final_sha256=digest(final_path),
                           initial_tensor_counts=counts, initial_actor_critic_wm_equal_to_bootstrap=True,
                           final_wm_and_normalization_equal_to_bootstrap=True,
                           no_additional_collection_or_wm_training=True))
        del first, final
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU verification unexpectedly initialized CUDA')
    result = dict(status='actual_tensor_identity_verified', bootstrap_sha256=bootstrap_sha,
                  bootstrap_iteration=500, replay_size=replay_size, replay_ptr=replay_ptr,
                  checks=checks, cpu_only=True, seconds=time.perf_counter() - started,
                  script_sha256=digest(Path(__file__)), campaign_sha256=digest(root / 'campaign.json'),
                  scope='Initial actor/critic/WM state and final frozen WM checked directly; bootstrap hash identifies the shared input replay file, not sampled rollout equality')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.raw, args.output), indent=2))
