import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.recover_evaluations import file_hash, prepare_job, recovery_plan
from experiments.ablation.run_suite import source_snapshot


class RecoveryTests(unittest.TestCase):
    def fixture(self, directory):
        repo = Path(directory)
        (repo / 'experiments/ablation').mkdir(parents=True)
        (repo / 'experiments/ablation/train.py').write_text('immutable = True\n')
        root = repo / 'logs/campaign'
        run = root / 'ppo_sim-seed0'
        run.mkdir(parents=True)
        (root / 'ppo_sim-seed0.json').write_text(json.dumps(dict(iterations=1, checkpoint_every=1)))
        (run / 'completion.json').write_text(json.dumps(dict(status='completed', iterations=1)))
        for iteration in (0, 1):
            (run / f'ckpt_{iteration:06d}.pt').write_bytes(f'checkpoint {iteration}'.encode())
        dest = run / 'ckpt_000000/repeat_000'
        dest.mkdir(parents=True)
        (dest / 'evaluation.json').write_text(json.dumps(dict(
            checkpoint_sha256=file_hash(run / 'ckpt_000000.pt'), evaluation_repeat_index=0, evaluation_repeat_count=2)))
        (dest / 'per_clip.csv').write_text('motion,normalized_return\nwalk,0.1\n')
        commands = [dict(command=['python', '-m', 'experiments.ablation.train', '--output', str(run)],
                         returncode=0, seconds=30.),
                    dict(command=['python', '-m', 'experiments.ablation.evaluate', '--checkpoint', str(run / 'ckpt_000000.pt'),
                                  '--output', str(dest), '--repeat-index', '0', '--repeat-count', '2'], returncode=0, seconds=10.)]
        campaign = dict(status='stopped', commands=commands, actual_gpu_hours=40/3600,
                        source_hashes=source_snapshot(repo), args=dict(max_gpu_hours=1., seeds=[0], methods=['ppo_sim'], evaluation_repeats=2))
        (root / 'campaign.json').write_text(json.dumps(campaign))
        return repo, root, campaign

    def test_recovery_only_schedules_missing_evaluations_and_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, root, _ = self.fixture(directory)
            before = file_hash(root / 'campaign.json')
            campaign, jobs = recovery_plan(root, repo)
            self.assertEqual(len(jobs), 3)
            self.assertTrue(all('experiments.ablation.evaluate' in j['command'] for j in jobs))
            self.assertEqual(campaign['actual_gpu_hours'], 40/3600)
            self.assertEqual(file_hash(root / 'campaign.json'), before)

    def test_recovery_rejects_changed_source_unaccounted_outputs_and_checkpoint_mismatch(self):
        for failure in ('source', 'output', 'checkpoint'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                repo, root, _ = self.fixture(directory)
                if failure == 'source':
                    (repo / 'experiments/ablation/train.py').write_text('changed = True\n')
                elif failure == 'output':
                    (root / 'ppo_sim-seed0/ckpt_000001/repeat_000').mkdir(parents=True)
                else:
                    (root / 'ppo_sim-seed0/ckpt_000000.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    recovery_plan(root, repo)

    def test_recovery_never_restarts_failed_training_or_increases_budget(self):
        for failure in ('running', 'subprocess', 'incomplete', 'budget', 'ledger'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                repo, root, campaign = self.fixture(directory)
                if failure == 'running':
                    campaign['status'] = 'running'
                elif failure == 'subprocess':
                    campaign['commands'][0]['returncode'] = 1
                elif failure == 'incomplete':
                    (root / 'ppo_sim-seed0/completion.json').write_text(json.dumps(dict(status='budget_reached', iterations=0)))
                elif failure == 'budget':
                    campaign['args']['max_gpu_hours'] = campaign['actual_gpu_hours']
                else:
                    campaign['actual_gpu_hours'] = 0
                (root / 'campaign.json').write_text(json.dumps(campaign))
                with self.assertRaises(ValueError):
                    recovery_plan(root, repo)

    def unstarted_fixture(self, directory):
        repo, root, campaign = self.fixture(directory)
        configs = repo / 'experiments/ablation/configs'
        configs.mkdir()
        config = dict(iterations=1, checkpoint_every=1, seed=0, subset='locomotion')
        for method in ('ppo_sim', 'ppo_wm_online'):
            (configs / f'formal-{method}.json').write_text(json.dumps(config))
        (root / 'ppo_sim-seed0.json').write_text(json.dumps(config))
        campaign['args'].update(phase='formal', subset='locomotion', methods=['ppo_sim', 'ppo_wm_online'])
        campaign['source_hashes'] = source_snapshot(repo)
        (root / 'campaign.json').write_text(json.dumps(campaign))
        return repo, root, campaign

    def test_unstarted_training_requires_explicit_option_and_preserves_original_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, root, original = self.unstarted_fixture(directory)
            with self.assertRaises(ValueError):
                recovery_plan(root, repo)
            campaign, jobs = recovery_plan(root, repo, include_unstarted_training=True)
            self.assertEqual(len(jobs), 8)  # Three old evaluations, one new training, four new evaluations.
            training = [job for job in jobs if 'training_config' in job]
            self.assertEqual(len(training), 1)
            self.assertEqual(campaign, original)
            self.assertFalse(training[0]['config_path'].exists())  # Dry planning has no writes.
            prepare_job(training[0])
            self.assertEqual(json.loads(training[0]['config_path'].read_text()), training[0]['training_config'])

    def test_unstarted_option_rejects_any_evidence_of_prior_training_or_config_changes(self):
        for artifact in ('directory', 'log', 'memory', 'config', 'old_config'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                repo, root, _ = self.unstarted_fixture(directory)
                if artifact == 'directory':
                    (root / 'ppo_wm_online-seed0').mkdir()
                elif artifact in ('log', 'memory'):
                    suffix = '.log' if artifact == 'log' else '.gpu-memory.csv'
                    (root / f'ppo_wm_online-seed0-train{suffix}').write_text('previous work')
                else:
                    method = 'ppo_sim' if artifact == 'old_config' else 'ppo_wm_online'
                    (root / f'{method}-seed0.json').write_text('{}')
                with self.assertRaises(ValueError):
                    recovery_plan(root, repo, include_unstarted_training=True)

    def test_job_inputs_are_rechecked_after_planning(self):
        for mutation in ('checkpoint', 'training_output', 'training_config'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                repo, root, _ = self.unstarted_fixture(directory)
                _, jobs = recovery_plan(root, repo, include_unstarted_training=True)
                if mutation == 'checkpoint':
                    job = jobs[0]
                    job['checkpoint'].write_bytes(b'changed after planning')
                else:
                    job = next(job for job in jobs if 'training_config' in job)
                    if mutation == 'training_output':
                        job['cache'].parent.mkdir()
                    else:
                        job['config_path'].write_text('{}')
                with self.assertRaises(ValueError):
                    prepare_job(job)

    def test_unstarted_evaluation_cannot_run_before_training_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, root, _ = self.unstarted_fixture(directory)
            _, jobs = recovery_plan(root, repo, include_unstarted_training=True)
            job = jobs[-1]
            run = job['checkpoint'].parent
            run.mkdir()
            job['checkpoint'].write_bytes(b'partial checkpoint')
            (run / 'completion.json').write_text(json.dumps(dict(status='budget_reached', iterations=0)))
            with self.assertRaises(ValueError):
                prepare_job(job)

    def test_frozen_recovery_pins_the_existing_bootstrap_and_its_paid_pretraining(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, root, campaign = self.unstarted_fixture(directory)
            config = dict(iterations=1, checkpoint_every=1, seed=0, subset='locomotion')
            for method in ('pretrain', 'bptt_wm_frozen'):
                (repo / f'experiments/ablation/configs/formal-{method}.json').write_text(json.dumps(config))
            run = root / 'pretrain-seed0'
            run.mkdir()
            bootstrap = run / 'bootstrap.pt'
            bootstrap.write_bytes(b'original shared world model and replay')
            (run / 'completion.json').write_text(json.dumps(dict(status='completed', iterations=1)))
            (root / 'pretrain-seed0.json').write_text(json.dumps(config))
            campaign['commands'].insert(0, dict(command=['python', '-m', 'experiments.ablation.train', '--output', str(run)],
                                                 returncode=0, seconds=10.))
            campaign['actual_gpu_hours'] += 10/3600
            campaign['args']['methods'] = ['pretrain', 'ppo_sim', 'bptt_wm_frozen']
            campaign['source_hashes'] = source_snapshot(repo)
            (root / 'campaign.json').write_text(json.dumps(campaign))
            _, jobs = recovery_plan(root, repo, include_unstarted_training=True)
            job = next(job for job in jobs if 'training_config' in job)
            self.assertEqual(job['bootstrap_sha256'], file_hash(bootstrap))
            bootstrap.write_bytes(b'different world model')
            with self.assertRaises(ValueError):
                prepare_job(job)
            (run / 'completion.json').write_text(json.dumps(dict(status='completed', iterations=0)))
            with self.assertRaises(ValueError):
                recovery_plan(root, repo, include_unstarted_training=True)


if __name__ == '__main__':
    unittest.main()
