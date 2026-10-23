import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.forecast import calibrated_forecast


class MeasuredForecastTests(unittest.TestCase):
    def run_fixture(self, root, method, *, actor_updates=20):
        run = root / method / f'{method}-seed0'
        run.mkdir(parents=True)
        cfg = dict(method=method, subset='locomotion', num_envs=4096, pi_batch=4096,
                   horizon=32, hist=10, goal_frames=10, wm_window=8, wm_batch=2048,
                   wm_batches=10, collect_steps=2.5, amp='none', initial_std=.1,
                   gamma=.99, lam=.95, ppo_epochs=5, minibatches=4,
                   actor_lr=2e-5 if method.startswith('ppo') else None, wm_only_iters=2,
                   warmup_steps=64, buffer_steps=1000, lr=2e-4)
        manifest = dict(config=cfg, gpu='A100', gpu_uuid='probe-gpu', motions={'walk': 'a'},
                        assets={'robot': 'b'}, packages={'physics': 'c'}, torch='test', cuda='test',
                        source_hashes={'supermimic/mdp.py': 'same'})
        (run / 'manifest.json').write_text(json.dumps(manifest))
        (run / 'completion.json').write_text(json.dumps(dict(status='completed', iterations=40)))
        rows = [dict(iteration=i, train_wall_seconds=100 + 2 * i, collection_seconds=i * .2,
                     wm_training_seconds=i * .5, imagined_rollout_seconds=i * .4,
                     policy_optimization_seconds=i * .8, actor_updates=actor_updates) for i in range(1, 41)]
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
