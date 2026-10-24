import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.forecast import FROZEN_METHODS, calibrated_forecast


class MeasuredForecastTests(unittest.TestCase):
    def run_fixture(self, root, method, *, actor_updates=20, bootstrap_run=None):
        run = root / method / f'{method}-seed0'
        run.mkdir(parents=True)
        cfg = dict(method=method, subset='locomotion', num_envs=4096, pi_batch=4096,
                   horizon=32, hist=10, goal_frames=10, wm_window=8, wm_batch=2048,
                   wm_batches=10, collect_steps=2.5, amp='none', initial_std=.1,
                   gamma=.99, lam=.95, ppo_epochs=5, minibatches=4,
                   actor_lr=2e-5 if method.startswith('ppo') else None, wm_only_iters=2,
                   warmup_steps=64, buffer_steps=1000, lr=2e-4,
                   compile=True, clip=.2, entropy_coef=0., target_kl=.02, lr_decay_iters=250000,
                   bootstrap=str(bootstrap_run / 'bootstrap.pt') if bootstrap_run else None)
        manifest = dict(config=cfg, gpu='A100', gpu_uuid='probe-gpu', motions={'walk': 'a'},
                        assets={'robot': 'b'}, packages={'physics': 'c'}, torch='test', cuda='test',
                        source_hashes={'supermimic/mdp.py': 'same'},
                        bootstrap_sha256='a' * 64 if bootstrap_run else None)
        rows = [dict(iteration=i, train_wall_seconds=100 + 2 * i, actual_run_seconds=100 + 2 * i,
                     simulator_steps=i * 100, collection_seconds=i * .2,
                     wm_training_seconds=i * .5, imagined_rollout_seconds=i * .4,
                     policy_optimization_seconds=i * .8, actor_updates=actor_updates) for i in range(1, 41)]
        if bootstrap_run:
            inherited = json.loads((bootstrap_run / 'completion.json').read_text())['ledger']
            for row in rows:
                row['train_wall_seconds'] += inherited['train_wall_seconds']
                for key in ('collection_seconds', 'wm_training_seconds', 'simulator_steps'):
                    row[key] = inherited[key]
        (run / 'manifest.json').write_text(json.dumps(manifest))
        (run / 'completion.json').write_text(json.dumps(dict(status='completed', iterations=40, ledger=rows[-1])))
        with (run / 'train.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        campaign = dict(commands=[dict(command=['python', '-m', 'experiments.ablation.train',
                                                '--output', str(run)], seconds=190., returncode=0,
                                       resources={'uuid': 'probe-gpu'},
                                       peak_device_memory_used_mib=10000.)])
        (run.parent / 'campaign.json').write_text(json.dumps(campaign))
        return run

    def forecast(self, root, *, repeats=3, updates=20):
        runs = {method: self.run_fixture(root, method, actor_updates=updates)
                for method in ('ppo_sim', 'ppo_wm_online', 'supermimic', 'pretrain')}
        return calibrated_forecast(runs, seeds=[1, 2, 3], gpu_uuid='target-gpu',
                                   iterations=100, checkpoint_every=25, evaluation_repeats=repeats,
                                   evaluation_seconds=600., disk_gib_per_seed=12.)

    def test_full_updates_repeats_pretrain_and_reserves_are_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.forecast(Path(directory))
        self.assertEqual(result['scope']['ppo_actor_lr'], 2e-5)
        self.assertEqual(result['scope']['evaluation_repeats'], 3)
        self.assertEqual(result['evaluation_processes'], 3 * 6 * 5 * 3)
        self.assertEqual(result['measurements']['ppo_sim']['training_seconds'], 310.)
        self.assertEqual(result['projected_disk_gib'], 36.)
        self.assertGreater(result['projected_campaign_gpu_hours'], 45.)
        self.assertEqual(result['gpu_uuid'], 'target-gpu')
        self.assertEqual(result['measurement_gpu_uuids'], ['probe-gpu'])
        self.assertEqual(result['pretrain_seconds_estimate'], 1110.)
        self.assertEqual(result['frozen_cost_basis'], 'inferred_online_policy_phases')

    def frozen_fixture(self, root):
        runs = {m: self.run_fixture(root, m) for m in ('ppo_sim', 'ppo_wm_online', 'supermimic', 'pretrain')}
        frozen = {m: self.run_fixture(root, m, bootstrap_run=runs['pretrain']) for m in FROZEN_METHODS}
        kwargs = dict(seeds=[1], gpu_uuid='target-gpu', iterations=100, checkpoint_every=25,
                      evaluation_repeats=3, evaluation_seconds=600., disk_gib_per_seed=12.)
        return runs, frozen, kwargs

    def test_direct_frozen_uses_own_process_and_charges_pretrain_once(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, frozen, kwargs = self.frozen_fixture(Path(directory))
            result = calibrated_forecast(runs, frozen_runs=frozen, **kwargs)
        self.assertEqual(result['frozen_cost_basis'], 'direct_process_allocation_plus_warm_intervals')
        self.assertEqual(result['frozen_training_seconds_estimates'], dict.fromkeys(FROZEN_METHODS, 310.))
        self.assertEqual(result['projected_training_gpu_hours'], (1110. + 6 * 310.) / 3600)
        self.assertEqual(result['evaluation_processes'], 6 * 5 * 3)
        self.assertEqual(result['measurements']['ppo_wm_frozen']['timing_field'], 'actual_run_seconds')
        self.assertEqual(set(result['sources']), set(runs) | set(frozen))

    def test_partial_frozen_measurements_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, frozen, kwargs = self.frozen_fixture(Path(directory))
            for partial in ({}, {'bptt_wm_frozen': frozen['bptt_wm_frozen']}):
                with self.subTest(methods=list(partial)), self.assertRaisesRegex(ValueError, 'all three frozen'):
                    calibrated_forecast(runs, frozen_runs=partial, **kwargs)

    def test_direct_frozen_identity_mismatch_rejected(self):
        for change, error in [('hash', 'same bootstrap'), ('bootstrap', 'completed bootstrap'),
                              ('actor_lr', 'Actor rates'), ('compile', 'configuration')]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                runs, frozen, kwargs = self.frozen_fixture(Path(directory))
                path = frozen['ppo_wm_frozen'] / 'manifest.json'
                manifest = json.loads(path.read_text())
                if change == 'hash':
                    manifest['bootstrap_sha256'] = 'b' * 64
                elif change == 'bootstrap':
                    manifest['config']['bootstrap'] = '/another/pretrain/bootstrap.pt'
                else:
                    manifest['config'][change] = 2e-4 if change == 'actor_lr' else False
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, error):
                    calibrated_forecast(runs, frozen_runs=frozen, **kwargs)

    def test_frozen_collection_updates_and_inherited_cost_changes_rejected(self):
        for field in ('collection_seconds', 'wm_training_seconds', 'simulator_steps', 'train_wall_seconds'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                runs, frozen, kwargs = self.frozen_fixture(Path(directory))
                path = frozen['bptt_wm_frozen'] / 'train.csv'
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                # An early change outside the last 30 warm intervals must also fail.
                rows[0][field] = str(float(rows[0][field]) + .1)
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, 'after pretrain|inherited timing'):
                    calibrated_forecast(runs, frozen_runs=frozen, **kwargs)

    def test_frozen_own_clock_and_allocation_must_be_valid(self):
        for change in ('nan', 'decreasing', 'short_allocation', 'completion_overrun'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                runs, frozen, kwargs = self.frozen_fixture(Path(directory))
                run = frozen['bptt_wm_frozen']
                if change in ('nan', 'decreasing'):
                    path = run / 'train.csv'
                    with path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    rows[0]['actual_run_seconds'] = 'nan' if change == 'nan' else '105'
                    with path.open('w') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                elif change == 'short_allocation':
                    path = run.parent / 'campaign.json'; data = json.loads(path.read_text())
                    data['commands'][0]['seconds'] = 170.
                    path.write_text(json.dumps(data))
                else:
                    path = run / 'completion.json'; data = json.loads(path.read_text())
                    data['ledger']['actual_run_seconds'] = 191.
                    path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'timing|allocation'):
                    calibrated_forecast(runs, frozen_runs=frozen, **kwargs)

    def test_actor_starved_measurement_cannot_qualify_full_update_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'full Actor'):
                self.forecast(Path(directory), updates=1)

    def test_incomplete_run_is_not_a_measured_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {m: self.run_fixture(root, m) for m in ('ppo_sim','ppo_wm_online','supermimic','pretrain')}
            (runs['ppo_sim']/'completion.json').write_text(json.dumps(dict(status='budget_exhausted', iterations=40)))
            with self.assertRaisesRegex(ValueError, 'completed'):
                calibrated_forecast(runs, seeds=[1], gpu_uuid='gpu', iterations=100,
                                     checkpoint_every=25, evaluation_repeats=3,
                                     evaluation_seconds=600., disk_gib_per_seed=12.)

    def test_legacy_uuid_requires_recorded_resource_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {m: self.run_fixture(root, m) for m in ('ppo_sim','ppo_wm_online','supermimic','pretrain')}
            path = runs['supermimic'] / 'manifest.json'
            manifest = json.loads(path.read_text()); del manifest['gpu_uuid']
            path.write_text(json.dumps(manifest))
            kwargs = dict(seeds=[1], gpu_uuid='gpu', iterations=100, checkpoint_every=25,
                          evaluation_repeats=3, evaluation_seconds=600., disk_gib_per_seed=12.)
            result = calibrated_forecast(runs, **kwargs)
            self.assertEqual(result['measurements']['supermimic']['gpu_uuid_source'], 'process_resource_record')
            path = runs['supermimic'].parent / 'campaign.json'
            campaign = json.loads(path.read_text()); del campaign['commands'][0]['resources']
            path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError, 'GPU identity'):
                calibrated_forecast(runs, **kwargs)
