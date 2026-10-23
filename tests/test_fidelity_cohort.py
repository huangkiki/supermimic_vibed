"""Synthetic numerical/bookkeeping fixtures, never experimental evidence."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from experiments.ablation.analyze_fidelity_cohort import (
    collect_fidelity_cohort, main, microsteps, seed_statistics, stage_associations,
)
from experiments.ablation.validate_fidelity import validate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    plan = json.loads((Path(__file__).parents[1] / 'experiments/ablation/configs/confirmatory-gradient-plan.json').read_text())
    plan.update(wm_iterations=[1, 2], collector_iteration=2, samples=2, directions=2,
                epsilons=[.01, .03], probe_gpu_uuid='GPU-probe')
    roots, evaluated, raw = [], [], []
    source = {'supermimic/mdp.py': 'fixed-core', 'experiments/ablation/fidelity.py': 'fixed-probe'}
    for seed in plan['validation_seeds']:
        campaign_root = root / f'probe-seed{seed}'
        roots.append(campaign_root)
        hashes = {i: hashlib.sha256(f'synthetic-{seed}-{i}'.encode()).hexdigest() for i in (1, 2)}
        bank_hash = hashlib.sha256(f'synthetic-bank-{seed}'.encode()).hexdigest()
        manifest = dict(config=dict(seed=seed, method='supermimic', subset='locomotion'),
                        gpu='test-gpu', gpu_uuid='GPU-probe', motions={'walk.csv': 'data'}, assets={'robot': 'asset'},
                        packages={}, torch='test', cuda='test', source_hashes=source)
        training_manifest = dict(manifest, gpu_uuid='GPU-training')
        for iteration in (1, 2):
            evaluated.append(dict(seed=seed, method='supermimic', iteration=iteration, checkpoint_sha256=hashes[iteration],
                                  physical_return=.1 * (seed + iteration), tracking_failure_rate=.5,
                                  evaluation_repeats=3, evaluation_return_std=.001))
            raw.append(dict(seed=seed, method='supermimic', manifest=training_manifest))
        commands = []
        for reward, horizon in itertools.product(plan['reward_objectives'], plan['gradient_horizons']):
            probe = campaign_root / f'gradient-components-{reward}-h{horizon}'
            args = {k: plan[k] for k in ('samples', 'prefix_steps', 'prefix_controller', 'directions', 'epsilons', 'contact_level', 'probe_seed')}
            args.update(reward_objective=reward, gradient_horizon=horizon)
            receipt = dict(args=args, manifest=manifest, collector_sha256=hashes[2], bank_sha256=bank_hash,
                           checkpoint_sha256={f'/synthetic/seed{seed}/ckpt_{i:06d}.pt': digest for i, digest in hashes.items()},
                           gpu='test-gpu', precision=plan['precision'], reset_protocol=plan['reset_protocol'])
            write_json(probe / 'probe.json', receipt)
            write_json(probe / 'bank.json', dict(bank_sha256=bank_hash, bank_path='/synthetic/not-a-file',
                       collector_sha256=hashes[2], samples=2, prefix_controller='policy', prefix_steps=16))
            write_csv(probe / 'bank-diagnostics.csv', [dict(sample=s, boundary_action_delta_l2=.1,
                      following_action_delta_l2_mean=.1) for s in range(2)])
            prediction = [dict(checkpoint_sha256=hashes[i], sample=s, horizon=h,
                               scaled_mse=(3 - i) * .01 * seed * h)
                          for i, s, h in itertools.product((1, 2), range(2), (1, 8, 16, 32))]
            write_csv(probe / 'prediction.csv', prediction)
            rows = []
            for i, s, objective, d, eps in itertools.product((1, 2), range(2), ('hard', 'soft'), range(3), plan['epsilons']):
                improvement = eps * seed if d == 2 else eps * (.25 if d == 0 else -.25)
                derivative = improvement / eps
                rows.append(dict(checkpoint_sha256=hashes[i], sample=s, gradient_horizon=horizon,
                                 reward_objective=reward, objective=objective, direction=d, epsilon=eps,
                                 direction_type='wm_gradient' if d == 2 else 'random_unit_l2',
                                 direction_l2_norm=1., wm_gradient_l2_norm=1., contact_group_level='body',
                                 contact_group='stable_contact' if s == 0 else 'switch_nearby',
                                 predicted_derivative=derivative * (-1 if seed == 2 else 1), fd_derivative=derivative,
                                 repeat_return_noise=1e-7, initial_state_max_drift=0., resolved='True',
                                 baseline_return=1., positive_return=1. + improvement, negative_return=1. - improvement,
                                 improvement=improvement))
            write_csv(probe / 'gradient.csv', rows)
            # Model a previous native verification receipt; no fake data are exported.
            native = validate(probe, write_receipt=False)
            native['bank_identity'] = 'sha256_verified'
            write_json(probe / 'validation.json', native)
            commands.append(dict(command=['python', '-m', 'experiments.ablation.fidelity', '--output', str(probe)],
                                 seconds=1., returncode=0))
        write_json(campaign_root / 'campaign.json', dict(status='completed', plan=dict(training_seed=seed, hard_gpu_hours=.1),
                   resources=dict(uuid='GPU-probe'), source_hashes=source, commands=commands, actual_gpu_hours=6 / 3600))
    return plan, roots, evaluated, raw


class FidelityCohortTests(unittest.TestCase):
    def test_seed_weight_is_independent_of_state_or_direction_count(self):
        rows = [dict(seed=s, horizon=32, score=value, directions=n) for s, value, n in
                [(1, 0., 1000), (2, 1., 1), (3, 1., 1)]]
        result = seed_statistics(rows, ('horizon',), ('score',), [1, 2, 3])[0]
        self.assertAlmostEqual(result['score_mean'], 2 / 3)
        self.assertAlmostEqual(result['score_std'], statistics.stdev([0., 1., 1.]))
        self.assertEqual(result['score_defined_seeds'], 3)

    def test_missing_duplicate_and_undefined_seed_metrics_do_not_become_false_precision(self):
        rows = [dict(seed=s, score=v) for s, v in [(1, .2), (2, None), (3, .4)]]
        result = seed_statistics(rows, (), ('score',), [1, 2, 3])[0]
        self.assertIsNone(result['score_mean'])
        self.assertIsNone(result['score_std'])
        self.assertEqual(result['score_defined_seeds'], 2)
        for invalid in (rows[:2], [*rows, rows[0]]):
            with self.assertRaises(ValueError):
                seed_statistics(invalid, (), ('score',), [1, 2, 3])

    def test_complete_cohort_links_bytes_counts_absent_contact_and_all_probe_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, roots, evaluations, raw = fixture(Path(directory))
            before = (roots[0] / 'gradient-components-full-h1/validation.json').read_bytes()
            paired, improvements, predictions, receipts = collect_fidelity_cohort(roots, evaluations, raw, plan)
            self.assertEqual(len(predictions), 3 * 2 * 4)  # Not multiplied by six repeated probes.
            self.assertEqual(len(receipts), 3)
            self.assertAlmostEqual(sum(r['actual_gpu_hours'] for r in receipts), 18 / 3600)
            self.assertEqual(before, (roots[0] / 'gradient-components-full-h1/validation.json').read_bytes())
            absent = [r for r in paired if r['contact_group'] == 'flight']
            self.assertTrue(absent)
            self.assertTrue(all(r['unique_states'] == 0 and r['full_sign_agreement'] is None for r in absent))
            step = next(r for r in improvements if r['seed'] == 1 and r['contact_group'] == 'all' and r['epsilon'] == .03)
            self.assertEqual(step['paired_states'], 2)
            self.assertAlmostEqual(step['gradient_minus_random'], .03)
            associations = stage_associations(paired, improvements, predictions, plan)
            self.assertEqual(len(associations), 9)
            self.assertAlmostEqual(associations[0]['mse_return_pearson'], -1.)
            self.assertIsNone(associations[0]['mse_sign_pearson'])  # Constant signs do not define r.

    def test_wrong_seed_gpu_checkpoint_and_unpaid_or_partial_probes_are_rejected(self):
        for mutation in ('seed', 'gpu', 'checkpoint', 'cost', 'duplicate_allocation', 'missing_probe', 'stale_validation', 'bank', 'mixed_source', 'precision'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                plan, roots, evaluations, raw = fixture(Path(directory))
                campaign_path = roots[0] / 'campaign.json'
                campaign = json.loads(campaign_path.read_text())
                probe = roots[0] / 'gradient-components-full-h1'
                if mutation == 'seed':
                    campaign['plan']['training_seed'] = 0
                elif mutation == 'gpu':
                    campaign['resources']['uuid'] = 'GPU-other'
                elif mutation == 'cost':
                    campaign['actual_gpu_hours'] += .1
                elif mutation == 'duplicate_allocation':
                    campaign['commands'].append(campaign['commands'][0])
                    campaign['actual_gpu_hours'] += 1 / 3600
                elif mutation == 'missing_probe':
                    (probe / 'probe.json').unlink()
                elif mutation == 'stale_validation':
                    p = probe / 'validation.json'
                    data = json.loads(p.read_text())
                    data['resolved_rows'] -= 1
                    write_json(p, data)
                elif mutation == 'mixed_source':
                    campaign['source_hashes']['experiments/ablation/fidelity.py'] = 'changed-probe'
                    for p in roots[0].glob('*/probe.json'):
                        data = json.loads(p.read_text())
                        data['manifest']['source_hashes']['experiments/ablation/fidelity.py'] = 'changed-probe'
                        write_json(p, data)
                else:
                    p = probe / 'probe.json'
                    data = json.loads(p.read_text())
                    if mutation == 'checkpoint':
                        data['collector_sha256'] = 'f' * 64
                    elif mutation == 'precision':
                        data['precision'] = 'different precision'
                    else:
                        data['bank_sha256'] = 'f' * 64
                    write_json(p, data)
                write_json(campaign_path, campaign)
                with self.assertRaises(ValueError):
                    collect_fidelity_cohort(roots, evaluations, raw, plan)

    def test_random_controls_and_gradient_updates_use_the_same_reproducible_states(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, roots, _, _ = fixture(Path(directory))
            probe = roots[0] / 'gradient-components-full-h1'
            receipt = json.loads((probe / 'probe.json').read_text())
            with (probe / 'gradient.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row['sample'] == '0' and row['direction'] == '0':
                    row['initial_state_max_drift'] = .01
            steps = microsteps(rows, receipt, 1)
            self.assertTrue(all(r['paired_states'] == 1 for r in steps if r['contact_group'] == 'all'))
            self.assertTrue(all(r['gradient_improvement'] is None for r in steps if r['contact_group'] == 'stable_contact'))
            rows[0]['direction_type'] = 'wm_gradient'
            with self.assertRaises(ValueError):
                microsteps(rows, receipt, 1)

    def test_missing_training_seed_cannot_be_replaced_by_repeated_probe_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, roots, evaluations, raw = fixture(Path(directory))
            for invalid in (roots[:2], [roots[0], roots[0], roots[1]]):
                with self.assertRaises(ValueError):
                    collect_fidelity_cohort(invalid, evaluations, raw, plan)
            paired, steps, predictions, _ = collect_fidelity_cohort(roots, evaluations, raw, plan)
            with self.assertRaises(ValueError):
                stage_associations(paired, steps, predictions[1:], plan)

    def test_implicit_legacy_actor_rate_is_normalized_without_accepting_a_changed_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, roots, evaluations, raw = fixture(Path(directory))
            for root in roots:
                for p in root.glob('*/probe.json'):
                    data = json.loads(p.read_text())
                    data['manifest']['config']['actor_lr'] = None
                    write_json(p, data)
            collect_fidelity_cohort(roots, evaluations, raw, plan)
            p = roots[0] / 'gradient-components-full-h1/probe.json'
            data = json.loads(p.read_text())
            data['manifest']['config']['actor_lr'] = .001
            write_json(p, data)
            with self.assertRaisesRegex(ValueError, 'config'):
                collect_fidelity_cohort(roots, evaluations, raw, plan)

    def test_cli_generates_all_tables_and_plots_only_after_cohort_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, probes, evaluated, raw = fixture(root)
            write_json(root / 'plan.json', plan)
            output = root / 'analysis'
            argv = ['analyze_fidelity_cohort', '--training-plan', str(root / 'plan.json'),
                    '--probe-plan', str(root / 'plan.json'), '--training-campaigns', 'validated-in-separate-tests',
                    '--probe-campaigns', *(str(p) for p in probes), '--output', str(output)]
            with patch('sys.argv', argv), patch('experiments.ablation.analyze_fidelity_cohort.collect_cohort', return_value=(evaluated, raw, [])):
                main()
            self.assertEqual(len(list(output.glob('*.csv'))), 8)
            self.assertEqual(len(list(output.glob('*.png'))), 3)
            self.assertTrue(all(p.stat().st_size > 5000 for p in output.glob('*.png')))
            receipt = json.loads((output / 'fidelity-cohort.json').read_text())
            self.assertEqual(len(receipt['probe_campaigns']), 3)
            with patch('sys.argv', argv), patch('experiments.ablation.analyze_fidelity_cohort.collect_cohort', side_effect=ValueError('incomplete real cohort')):
                before = (output / 'fidelity-cohort.json').read_bytes()
                with self.assertRaisesRegex(ValueError, 'incomplete real cohort'):
                    main()
                self.assertEqual(before, (output / 'fidelity-cohort.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
