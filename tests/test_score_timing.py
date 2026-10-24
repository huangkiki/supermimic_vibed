import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.score_timing import score_timing
from test_ablation_analysis import evaluation


class ScoreTimingTests(unittest.TestCase):
    def scenario(self, root, frozen=False, unused_pretrain=False):
        method = 'ppo_wm_frozen' if frozen else 'ppo_sim'
        run = root / f'{method}-seed0'
        run.mkdir()
        (run / 'completion.json').write_text(json.dumps(dict(iterations=2)))
        commands = []
        if frozen or unused_pretrain:
            pretrain = root / 'pretrain-seed0'
            pretrain.mkdir()
            (pretrain / 'completion.json').write_text(json.dumps(
                dict(iterations=1, ledger=dict(train_wall_seconds=30.))))
            commands.append(dict(command=['python', '-m', 'experiments.ablation.train',
                '--output', str(pretrain)], returncode=0, seconds=35.))
        commands.append(dict(command=['python', '-m', 'experiments.ablation.train',
            '--output', str(run)], returncode=0, seconds=240.))
        paths = []
        for iteration in (1, 2):
            for repeat in range(3):
                record = evaluation(iteration=iteration)
                record.update(method=method, evaluation_repeat_index=repeat, evaluation_repeat_count=3)
                record['evaluation_gpu_uuid'] = 'GPU-test'
                record['manifest']['gpu_uuid'] = 'GPU-test'
                path = run / f'eval-{iteration}-{repeat}.json'
                path.write_text(json.dumps(record))
                paths.append(path)
                commands.append(dict(command=['python', '-m', 'experiments.ablation.evaluate',
                    '--checkpoint', str(run / f'ckpt_{iteration:06d}.pt'),
                    '--repeat-index', str(repeat)], returncode=0, seconds=10.+repeat))
        campaign = root / 'campaign.json'
        campaign.write_text(json.dumps(dict(commands=commands)))
        return paths, campaign, commands

    def test_offline_scores_charge_full_training_before_the_first_result(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, _ = self.scenario(Path(directory))
            first, last = score_timing(paths, campaign)
            self.assertEqual(first['reconstructed_checkpoint_seconds'], 133.)
            self.assertEqual(first['score_available_allocation_seconds'], 273.)
            self.assertEqual(last['score_available_allocation_seconds'], 306.)
            self.assertEqual(last['score_available_allocation_seconds'], last['reconstructed_checkpoint_seconds'])

    def test_shared_pretraining_is_charged_once_for_frozen_method(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, _ = self.scenario(Path(directory), frozen=True)
            first, last = score_timing(paths, campaign)
            self.assertEqual(first['score_available_allocation_seconds'], 308.)
            self.assertEqual(last['score_available_allocation_seconds'], 341.)
            self.assertEqual(last['score_available_allocation_seconds'], last['reconstructed_checkpoint_seconds'])

    def test_unused_pretraining_is_not_charged_to_ppo_sim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, _ = self.scenario(Path(directory), unused_pretrain=True)
            first, last = score_timing(paths, campaign)
            self.assertEqual(first['score_available_allocation_seconds'], 273.)
            self.assertEqual(last['score_available_allocation_seconds'], 306.)

    def test_missing_physical_gpu_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, _ = self.scenario(Path(directory))
            for path in paths:
                record = json.loads(path.read_text())
                del record['evaluation_gpu_uuid']
                del record['manifest']['gpu_uuid']
                path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'GPU identity'):
                score_timing(paths, campaign)

    def test_missing_prior_checkpoint_or_repeat_cannot_hide_evaluation_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, _ = self.scenario(Path(directory))
            for incomplete in (paths[3:], paths[:-1]):
                with self.subTest(paths=incomplete), self.assertRaises(ValueError):
                    score_timing(incomplete, campaign)

    def test_unpaid_and_out_of_order_allocations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, campaign, commands = self.scenario(Path(directory))
            for mode in ('pending', 'before_training', 'reversed_checkpoints'):
                changed = json.loads(json.dumps(commands))
                if mode == 'pending':
                    changed[-1]['returncode'] = None
                    changed[-1]['seconds'] = None
                elif mode == 'before_training':
                    changed[0], changed[1] = changed[1], changed[0]
                else:
                    changed = changed[:1] + changed[4:] + changed[1:4]
                campaign.write_text(json.dumps(dict(commands=changed)))
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    score_timing(paths, campaign)


if __name__ == '__main__':
    unittest.main()
