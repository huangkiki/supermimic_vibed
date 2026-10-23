import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.ablation.continue_seeds import completed_cost, dependency_status, main, process_identity
from experiments.ablation.run_suite import METHOD_ORDER, campaign_scope


def write_json(path, value):
    path.write_text(json.dumps(value))


def fake_process(root, ticks=100, state='S', script='/runtime/seed1.py'):
    path = root / '123'
    path.mkdir(exist_ok=True)
    (path / 'stat').write_text('123 (python worker) ' + ' '.join([state, *(['0'] * 18), str(ticks), '0']))
    (path / 'cmdline').write_bytes(b'python3\0' + script.encode() + b'\0')
    return process_identity(123, root) if state not in ('Z', 'X') else None


class SeedContinuationTests(unittest.TestCase):
    def test_live_handle_is_bound_to_argv_and_start_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = fake_process(root)
            state = root / 'queue.json'
            write_json(state, dict(status='waiting'))
            self.assertEqual(dependency_status(state, identity, Path('/runtime/seed1.py'), root)['status'], 'waiting')
            fake_process(root, ticks=101)
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                dependency_status(state, identity, Path('/runtime/seed1.py'), root)

    def test_partial_status_write_does_not_mean_process_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = fake_process(root)
            state = root / 'queue.json'
            state.write_text('{"status":')
            self.assertEqual(dependency_status(state, identity, Path('/runtime/seed1.py'), root)['status'], 'partial_status_write')
            (root / '123/stat').unlink()
            with self.assertRaises(FileNotFoundError):
                dependency_status(state, identity, Path('/runtime/seed1.py'), root)

    def test_terminal_failure_and_zombie_do_not_launch_followup_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = fake_process(root)
            state = root / 'queue.json'
            write_json(state, dict(status='stopped', error='budget exhausted'))
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                dependency_status(state, identity, Path('/runtime/seed1.py'), root)
            fake_process(root, state='Z')
            with self.assertRaisesRegex(RuntimeError, 'terminal'):
                process_identity(123, root)

    def test_completed_receipt_does_not_need_a_still_running_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'queue.json'
            write_json(state, dict(status='completed'))
            self.assertEqual(dependency_status(state, None, Path('/runtime/seed1.py'), root)['status'], 'completed')
            write_json(state, dict(status='running'))
            with self.assertRaisesRegex(RuntimeError, 'changed back'):
                dependency_status(state, None, Path('/runtime/seed1.py'), root)

    def completed_fixture(self, root):
        plan = dict(methods=list(METHOD_ORDER[1:]), iterations=2, checkpoint_every=1, ppo_actor_lr=.00002,
                    evaluation_repeats=3, subset='locomotion', target_gpu_uuid='GPU-fixed',
                    campaign_gpu_hours_cap_per_seed=32.)
        commands = [dict(seconds=10., returncode=0) for _ in range(7 + 6 * 3 * 3)]
        spend = len(commands) * 10 / 3600
        sources = {'immutable': 'sha'}
        campaign = dict(status='completed', args=dict(seeds=[1], phase='formal', subset='locomotion', max_gpu_hours=32.),
                        scope=campaign_scope(METHOD_ORDER, 2, 1, .00002, 3), resources=dict(uuid='GPU-fixed'),
                        source_hashes=sources, commands=commands, actual_gpu_hours=spend)
        validation = dict(actual_gpu_hours=spend, evaluation_processes=54,
                          checks=[dict(method=m, seed=1, training_and_full_physical_evaluation_valid=True) for m in METHOD_ORDER])
        write_json(root / 'campaign.json', campaign)
        write_json(root / 'validation.json', validation)
        return plan, sources, campaign, validation, spend

    def test_next_seed_requires_the_original_cap_identity_cost_and_native_qualification(self):
        for mutation in ('none', 'seed', 'source', 'cap', 'ledger', 'failed', 'partial', 'validation', 'repeats'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, sources, campaign, validation, spend = self.completed_fixture(root)
                if mutation == 'seed':
                    campaign['args']['seeds'] = [0]
                elif mutation == 'source':
                    campaign['source_hashes'] = {'changed': 'sha'}
                elif mutation == 'cap':
                    campaign['args']['max_gpu_hours'] = 33.
                elif mutation == 'ledger':
                    campaign['actual_gpu_hours'] = 0.
                elif mutation == 'failed':
                    campaign['commands'][0]['returncode'] = 1
                elif mutation == 'partial':
                    campaign['commands'].pop()
                elif mutation == 'validation':
                    validation['checks'][0]['training_and_full_physical_evaluation_valid'] = False
                elif mutation == 'repeats':
                    validation['evaluation_processes'] = 18
                write_json(root / 'campaign.json', campaign)
                write_json(root / 'validation.json', validation)
                if mutation == 'none':
                    self.assertAlmostEqual(completed_cost(root, 1, plan, sources), spend)
                else:
                    with self.assertRaises(ValueError):
                        completed_cost(root, 1, plan, sources)

    def test_insufficient_fresh_disk_preflight_cannot_start_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = root / 'execution'
            campaign = execution / 'logs/ablation/confirmatory-locomotion-seed1-v2'
            campaign.mkdir(parents=True)
            plan, _, record, _, _ = self.completed_fixture(campaign)
            plan.update(validation_seeds=[1, 2, 3], development_seeds=[0], campaign_gpu_hours_cap_total=96.)
            record['source_hashes'] = {}
            write_json(campaign / 'campaign.json', record)
            smoke = execution / 'logs/ablation/all40-smoke-v1'
            smoke.mkdir()
            write_json(smoke / 'campaign.json', dict(status='completed', source_hashes={}))
            write_json(root / 'plan.json', plan)
            forecast = dict(seeds=[1], scope=record['scope'], gpu='test-gpu', gpu_uuid='GPU-fixed', subset='locomotion',
                            projected_campaign_gpu_hours=1., projected_peak_memory_mib=1000.,
                            projected_disk_gib=12., minimum_disk_reserve_gib=12.)
            write_json(root / 'forecast.json', forecast)
            write_json(root / 'predecessor.json', dict(status='completed'))
            (root / 'predecessor.py').write_text('# Completed identified queue\n')
            output = root / 'continuation'
            argv = ['continue_seeds', '--execution-root', str(execution), '--plan', str(root / 'plan.json'),
                    '--forecast', str(root / 'forecast.json'), '--predecessor-queue', str(root / 'predecessor.json'),
                    '--predecessor-pid-file', str(root / 'no-longer-live.pid'),
                    '--predecessor-script', str(root / 'predecessor.py'), '--gpu', '0', '--output', str(output)]
            resources = dict(name='test-gpu', uuid='GPU-fixed', memory_free_mib='40000')
            with patch('sys.argv', argv), patch('experiments.ablation.continue_seeds.wait_for_idle_gpu', return_value=(resources, [])), \
                    patch('experiments.ablation.continue_seeds.shutil.disk_usage', return_value=SimpleNamespace(free=10 * 2**30)), \
                    patch('experiments.ablation.continue_seeds.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'disk'):
                    main()
            launch.assert_not_called()
            state = json.loads((output / 'queue.json').read_text())
            self.assertEqual(state['status'], 'stopped')
            self.assertEqual(state['completed_seeds'], [1])
            self.assertFalse((execution / 'logs/ablation/confirmatory-locomotion-seed2-v2').exists())


if __name__ == '__main__':
    unittest.main()
