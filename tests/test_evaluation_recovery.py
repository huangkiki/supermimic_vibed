import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.recover_evaluations import file_hash, recovery_plan
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


if __name__ == '__main__':
    unittest.main()
