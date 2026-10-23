import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ablation.analyze_cohort import collect_cohort, main, target_statistics


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    """Synthetic bookkeeping only; no fixtures are published as experiment results."""
    methods = ['ppo_sim', 'ppo_wm_online', 'supermimic', 'ppo_wm_frozen_hard', 'ppo_wm_frozen', 'bptt_wm_frozen']
    plan = dict(validation_seeds=[1, 2, 3], development_seeds=[0], methods=methods, subset='locomotion',
                iterations=2, checkpoint_every=1, evaluation_repeats=3, pretrain_iters=1,
                num_envs=4, horizon=8, wm_window=1, wm_only_iters=0, initial_std=.1,
                ppo_actor_lr=.00002, other_actor_wm_critic_lr=.0002, target_gpu_uuid='GPU-test',
                campaign_gpu_hours_cap_per_seed=1., campaign_gpu_hours_cap_total=3.,
                primary_endpoint=dict(return_threshold=.8, max_tracking_failure_rate=.2), secondary_return_thresholds=[.5, .7])
    paths = []
    for seed in plan['validation_seeds']:
        output = root / f'campaign-seed{seed}'
        commands = []
        for method in ['pretrain', *methods]:
            run = output / f'{method}-seed{seed}'
            frozen = '_frozen' in method
            iterations = 1 if method == 'pretrain' else 2
            cfg = dict(method=method, seed=seed, subset='locomotion', iterations=iterations, checkpoint_every=1,
                       num_envs=4, pi_batch=4, horizon=8, hist=10, goal_frames=10, wm_window=1, wm_only_iters=0,
                       initial_std=.1, lr=.0002, actor_lr=.00002 if method.startswith('ppo') else None,
                       gamma=.99, lam=.95, amp='none',
                       bootstrap=str(output / f'pretrain-seed{seed}/bootstrap.pt') if frozen else None)
            manifest = dict(config=cfg, motions={'walk.csv': 'fixed-data'}, assets={'robot': 'fixed-asset'},
                            gpu='test-gpu', gpu_uuid='GPU-test', torch='test', cuda='test', packages={},
                            source_hashes={'supermimic/mdp.py': 'fixed-source'},
                            bootstrap_sha256=f'shared-within-seed{seed}' if frozen else None)
            write_json(run / 'manifest.json', manifest)
            write_json(run / 'completion.json', dict(status='completed', iterations=iterations,
                                                     ledger=dict(train_wall_seconds=10. if method == 'pretrain' else 80.)))
            write_csv(run / 'train.csv', [dict(iteration=i) for i in range(1, iterations + 1)])
            commands.append(dict(command=['python', '-m', 'experiments.ablation.train', '--output', str(run)],
                                 seconds=10. if method == 'pretrain' else 100., returncode=0))
            if method == 'pretrain':
                continue
            for iteration in (0, 1, 2):
                for repeat in range(3):
                    score = (.1, .65, .8 + .05 * (seed - 1))[iteration] + (repeat - 1) * .01
                    failed = iteration < 2
                    dest = run / f'ckpt_{iteration:06d}/repeat_{repeat:03d}'
                    record = dict(method=method, seed=seed, iteration=iteration, manifest=manifest,
                                  evaluation_manifest=manifest, protocol='fixed-physical-evaluation',
                                  physical_return=score, tracking_failure_rate=float(failed), total_clips=1,
                                  failed_clips=int(failed), evaluation_repeat_index=repeat, evaluation_repeat_count=3,
                                  checkpoint_sha256=f'{method}-{seed}-{iteration}', evaluation_gpu='test-gpu',
                                  evaluation_gpu_uuid='GPU-test', evaluation_seconds=4., evaluation_simulator_steps=10,
                                  train_ledger=dict(train_wall_seconds=40. * iteration + (10. if frozen else 0),
                                                    simulator_steps=64 if frozen else 64 + 100 * iteration,
                                                    imagined_steps=0 if method == 'ppo_sim' else 100 * iteration,
                                                    collection_seconds=5. if frozen else 10. * iteration,
                                                    wm_training_seconds=5. if frozen else 10. * iteration,
                                                    imagined_rollout_seconds=5. * iteration,
                                                    policy_optimization_seconds=10. * iteration))
                    write_json(dest / 'evaluation.json', record)
                    write_csv(dest / 'per_clip.csv', [dict(motion='walk', normalized_return=score, failed=str(failed),
                                                         steps=5 if failed else 10, reference_steps=10)])
                    commands.append(dict(command=['python', '-m', 'experiments.ablation.evaluate', '--checkpoint',
                                                 str(run / f'ckpt_{iteration:06d}.pt'), '--repeat-index', str(repeat)],
                                         seconds=5., returncode=0))
        campaign = dict(status='completed', args=dict(phase='formal', subset='locomotion', methods=['pretrain', *methods],
                        seeds=[seed], iterations=2, checkpoint_every=1, evaluation_repeats=3, ppo_actor_lr=.00002,
                        max_gpu_hours=1.), commands=commands, actual_gpu_hours=sum(c['seconds'] for c in commands) / 3600,
                        resources=dict(uuid='GPU-test'), source_hashes={'supermimic/mdp.py': 'fixed-source', 'immutable-execution': 'fixed'})
        path = output / 'campaign.json'
        write_json(path, campaign)
        write_json(output / 'validation.json', dict(actual_gpu_hours=campaign['actual_gpu_hours'],
                   checks=[dict(method=method, seed=seed, training_and_full_physical_evaluation_valid=True)
                           for method in ['pretrain', *methods]]))
        paths.append(path)
    return plan, paths


class CohortTests(unittest.TestCase):
    def test_full_cohort_preserves_seed_count_and_charges_pretrain_per_standalone_method(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = fixture(Path(directory))
            rows, raw, receipts = collect_cohort(paths, plan)
            self.assertEqual((len(rows), len(raw), len(receipts)), (54, 162, 3))
            final = [r for r in rows if r['iteration'] == 2]
            # Three frozen methods pay 10 seconds each; the physical campaign paid it once per seed.
            self.assertAlmostEqual(sum(r['total_gpu_hours'] for r in final)
                                   - sum(r['actual_gpu_hours'] for r in receipts), 3 * 20 / 3600)

    def test_partial_duplicate_or_development_seed_cannot_be_published_as_confirmatory(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = fixture(Path(directory))
            for supplied in (paths[:2], [*paths, paths[0]]):
                with self.assertRaises(ValueError):
                    collect_cohort(supplied, plan)
            changed = copy.deepcopy(plan)
            changed['development_seeds'] = [1]
            with self.assertRaises(ValueError):
                collect_cohort(paths, changed)

    def test_incomplete_run_ledger_gpu_and_source_mismatch_are_rejected(self):
        for mutation in ('status', 'budget', 'nonfinite_budget', 'gpu', 'source', 'duplicate_cost', 'missing_repeat', 'clip_score', 'short_success', 'qualification'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                plan, paths = fixture(Path(directory))
                p = paths[0]
                c = json.loads(p.read_text())
                if mutation == 'status':
                    c['status'] = 'stopped'
                elif mutation == 'budget':
                    c['actual_gpu_hours'] += .1
                elif mutation == 'nonfinite_budget':
                    c['args']['max_gpu_hours'] = float('nan')
                elif mutation == 'gpu':
                    c['resources']['uuid'] = 'another-GPU'
                elif mutation == 'source':
                    c['source_hashes']['immutable-execution'] = 'changed'
                elif mutation == 'duplicate_cost':
                    c['commands'].append(c['commands'][-1])
                    c['actual_gpu_hours'] += c['commands'][-1]['seconds'] / 3600
                elif mutation == 'missing_repeat':
                    next(p.parent.rglob('evaluation.json')).unlink()
                elif mutation == 'qualification':
                    (p.parent / 'validation.json').unlink()
                elif mutation == 'short_success':
                    clip = next(p.parent.glob('ppo_sim-seed1/ckpt_000002/repeat_000/per_clip.csv'))
                    text = clip.read_text().replace('False,10,10', 'False,9,10')
                    clip.write_text(text)
                else:
                    clip = next(p.parent.rglob('per_clip.csv'))
                    clip.write_text('motion,normalized_return,failed,steps,reference_steps\nwalk,99,True,5,10\n')
                write_json(p, c)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    collect_cohort(paths, plan)

    def test_different_frozen_bootstrap_and_changed_actor_rate_are_rejected(self):
        for mutation in ('bootstrap', 'actor_lr'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                plan, paths = fixture(Path(directory))
                run = paths[0].parent / 'ppo_wm_frozen-seed1'
                manifest = json.loads((run / 'manifest.json').read_text())
                if mutation == 'bootstrap':
                    manifest['bootstrap_sha256'] = 'different-model'
                else:
                    manifest['config']['actor_lr'] = .9
                write_json(run / 'manifest.json', manifest)
                for p in run.rglob('evaluation.json'):
                    r = json.loads(p.read_text());r['manifest'] = manifest
                    write_json(p, r)
                with self.assertRaises(ValueError):
                    collect_cohort(paths, plan)

    def test_nonattainment_is_censored_and_suppresses_uncensored_mean_and_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = fixture(Path(directory))
            rows, _, _ = collect_cohort(paths, plan)
            for row in rows:
                if row['method'] == 'ppo_wm_online' and row['seed'] == 3:
                    row['physical_return'] = .79
            observed, summary, paired = target_statistics(rows, plan)
            item = next(r for r in summary if r['method'] == 'ppo_wm_online' and r['return_threshold'] == .8)
            self.assertEqual((item['attained'], item['seeds'], item['censored']), (2, 3, 1))
            self.assertIsNone(item['total_gpu_hours_mean'])
            self.assertIsNone(item['end_to_end_seconds_std'])
            pair = next(r for r in paired if r['candidate'] == 'ppo_wm_online' and r['return_threshold'] == .8)
            self.assertIsNone(pair['paired_gpu_cost_ratio_mean'])
            missing = next(r for r in observed if r['method'] == 'ppo_wm_online' and r['seed'] == 3 and r['return_threshold'] == .8)
            self.assertFalse(missing['attained'])
            self.assertEqual(missing['observed_iteration'], 2)

    def test_first_observed_target_requires_both_return_and_failure_criteria(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, paths = fixture(Path(directory))
            rows, _, _ = collect_cohort(paths, plan)
            for row in rows:
                if row['iteration'] == 1:
                    row['physical_return'] = .99  # High reward alone cannot override tracking failures.
            observed, summary, paired = target_statistics(rows, plan)
            self.assertTrue(all(r['observed_iteration'] == 2 for r in observed))
            self.assertTrue(all(r['attained'] == 3 for r in summary))
            self.assertTrue(all(r['paired_gpu_cost_ratio_mean'] == 1. for r in paired))

    def test_cli_generates_seed_statistics_and_censoring_tables_from_complete_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, paths = fixture(root)
            write_json(root / 'plan.json', plan)
            output = root / 'analysis'
            with patch('sys.argv', ['analyze_cohort', '--plan', str(root / 'plan.json'), '--campaigns',
                                    *map(str, paths), '--output', str(output)]):
                main()
            with (output / 'final-seed-statistics.csv').open() as stream:
                final = list(csv.DictReader(stream))
            self.assertEqual(len(final), 6)
            self.assertTrue(all(r['seeds'] == '3' for r in final))
            self.assertAlmostEqual(float(final[0]['physical_return_mean']), .85)
            self.assertAlmostEqual(float(final[0]['physical_return_std']), .05)
            self.assertTrue((output / 'return-compute.png').is_file())
            self.assertTrue((output / 'paired-target-costs.csv').is_file())
            receipt = json.loads((output / 'cohort.json').read_text())
            self.assertEqual(len(receipt['campaigns']), 3)


if __name__ == '__main__':
    unittest.main()
