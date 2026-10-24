import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.ablation.run_fidelity_cohort import commands_for_seed, execute_probe, main, preflight, validate_plan
from test_cohort_analysis import fixture, write_json


MODULE = 'experiments.ablation.run_fidelity_cohort.'
PLAN = Path(__file__).parents[1] / 'experiments/ablation/configs/confirmatory-gradient-plan.json'


class FidelityRunnerTests(unittest.TestCase):
    def fixture(self, root):
        training, campaigns = fixture(root / 'training')
        plan = json.loads(PLAN.read_text())
        plan.update(collector_iteration=2, wm_iterations=[1, 2], samples=2, directions=1,
                    probe_gpu_uuid='GPU-probe')
        for campaign in campaigns:
            run = campaign.parent / f"supermimic-seed{json.loads(campaign.read_text())['args']['seeds'][0]}"
            for stage in (1, 2):
                checkpoint = run / f'ckpt_{stage:06d}.pt'
                checkpoint.write_bytes(f'{run.name}-{stage}'.encode())
                sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                for path in (run / checkpoint.stem).glob('*/evaluation.json'):
                    record = json.loads(path.read_text()); record['checkpoint_sha256'] = sha
                    write_json(path, record)
        write_json(root / 'training-plan.json', training)
        write_json(root / 'probe-plan.json', plan)
        output = root / 'probes'
        argv = ['run_fidelity_cohort', '--training-plan', str(root / 'training-plan.json'),
                '--probe-plan', str(root / 'probe-plan.json'), '--training-campaigns', *map(str, campaigns),
                '--gpu', '2', '--output', str(output)]
        return training, plan, campaigns, output, argv

    def test_commands_share_bank_within_seed_and_never_across_seeds(self):
        plan = json.loads(PLAN.read_text())
        a = commands_for_seed(Path('/trained/seed1'), Path('/probe/seed1'), plan, '/python')
        b = commands_for_seed(Path('/trained/seed2'), Path('/probe/seed2'), plan, '/python')
        self.assertEqual(len(a), 6)
        self.assertNotIn('--bank', a[0][1])
        for dest, command in a[1:]:
            self.assertEqual(command[command.index('--bank') + 1], str(a[0][0] / 'heldout-bank.pt'))
            self.assertEqual(command[command.index('--contact-level') + 1], 'body')
        self.assertNotEqual(a[1][1][-1], b[1][1][-1])

    def test_invalid_registered_budget_or_design_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            training, plan, *_ = self.fixture(Path(directory))
            for key, value in [('probe_gpu_hours_cap_total', .1), ('projected_gpu_hours_per_seed', .4),
                               ('minimum_free_memory_mib', float('nan')), ('samples', 0),
                               ('epsilons', [.01, .01]), ('precision', 'fp16')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_plan(training, {**plan, key: value})

    def test_real_cohort_gate_and_checkpoint_bytes_precede_gpu_queries(self):
        for mutation in ('incomplete', 'bytes'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, campaigns, output, argv = self.fixture(root)
                if mutation == 'incomplete':
                    data = json.loads(campaigns[0].read_text()); data['status'] = 'running'
                    write_json(campaigns[0], data)
                else:
                    (campaigns[0].parent / 'supermimic-seed1/ckpt_000002.pt').write_bytes(b'changed')
                with patch('sys.argv', argv), patch(MODULE + 'source_snapshot', return_value={'supermimic/mdp.py': 'fixed-source'}), \
                        patch(MODULE + 'wait_for_idle_gpu') as gpu, patch(MODULE + 'subprocess.Popen') as launch:
                    with self.assertRaises(ValueError):
                        main()
                gpu.assert_not_called(); launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_dry_run_validates_complete_cohort_without_outputs_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output, argv = self.fixture(Path(directory))
            with patch('sys.argv', [*argv, '--dry-run']), patch(MODULE + 'source_snapshot', return_value={'supermimic/mdp.py': 'fixed-source'}), \
                    patch(MODULE + 'wait_for_idle_gpu') as gpu, patch(MODULE + 'subprocess.Popen') as launch, patch('builtins.print') as emit:
                main()
            commands = json.loads(emit.call_args.args[0])['commands']
            self.assertEqual(set(commands), {'1', '2', '3'})
            self.assertTrue(all(len(c) == 6 for c in commands.values()))
            gpu.assert_not_called(); launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_fresh_preflight_rejects_memory_and_disk_shortfall(self):
        plan = json.loads(PLAN.read_text())
        for free_mib, free_gib in ((4096, 100), (40000, 20), (float('nan'), 100)):
            with self.subTest(memory=free_mib, disk=free_gib), \
                    patch(MODULE + 'wait_for_idle_gpu', return_value=({'name': 'A100', 'memory_free_mib': free_mib}, [])), \
                    patch(MODULE + 'shutil.disk_usage', return_value=SimpleNamespace(free=free_gib * 2**30)):
                with self.assertRaises(ValueError):
                    preflight(2, plan, 'A100', Path('/output'), 3, 600)

    def test_serial_suite_costs_and_fail_stop_preserve_completed_allocations(self):
        for failure in (None, 'process', 'changed_checkpoint', 'changed_bank', 'validation', 'budget'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, campaigns, output, argv = self.fixture(root)
                if failure == 'budget':
                    path = root / 'probe-plan.json'; plan = json.loads(path.read_text())
                    plan.update(probe_gpu_hours_cap_per_seed=.02, probe_gpu_hours_cap_total=.011,
                                projected_gpu_hours_per_seed=.003)
                    write_json(path, plan)
                calls = []
                def execute(record, log, **kwargs):
                    calls.append(record['command'])
                    record.update(returncode=7 if failure == 'process' else 0, seconds=2.)
                    kwargs['persist']()
                    if failure == 'process':
                        raise RuntimeError('probe failed')
                    command = record['command']
                    probe = Path(command[command.index('--output') + 1])
                    probe.mkdir()
                    bank = Path(command[command.index('--bank') + 1]) if '--bank' in command else probe / 'heldout-bank.pt'
                    if not bank.exists():
                        bank.write_bytes(str(probe.parent).encode())
                    write_json(probe / 'probe.json', dict(bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
                    if failure == 'changed_checkpoint':
                        (campaigns[0].parent / 'supermimic-seed1/ckpt_000002.pt').write_bytes(b'changed')
                    if failure == 'changed_bank':
                        bank.write_bytes(b'changed')
                validator = Mock(return_value={'bank_identity': 'sha256_verified'})
                if failure == 'validation':
                    validator.side_effect = ValueError('invalid recording')
                with patch('sys.argv', argv), patch(MODULE + 'source_snapshot', return_value={'supermimic/mdp.py': 'fixed-source'}), \
                        patch(MODULE + 'preflight', return_value=({'uuid': 'GPU-probe'}, [])), \
                        patch(MODULE + 'execute_probe', side_effect=execute), patch(MODULE + 'validate', validator), \
                        patch(MODULE + 'collect_seed'), patch(MODULE + 'collect_fidelity_cohort'):
                    if failure:
                        with self.assertRaises((ValueError, RuntimeError, TimeoutError)):
                            main()
                    else:
                        main()
                suite = json.loads((output / 'suite.json').read_text())
                self.assertEqual(len(calls), 15 if failure == 'budget' else 1 if failure else 18)
                self.assertEqual(suite['status'], 'stopped' if failure else 'completed')
                self.assertAlmostEqual(suite['actual_gpu_hours'], len(calls) * 2 / 3600)
                for seed, campaign in suite['seeds'].items():
                    self.assertEqual(campaign['plan']['training_seed'], int(seed))
                    self.assertEqual(campaign['plan']['hard_gpu_hours'], .02 if failure == 'budget' else .3)
                if failure == 'budget':
                    self.assertEqual(suite['seeds']['1']['status'], 'completed')
                    self.assertEqual(suite['seeds']['2']['status'], 'completed')

    def test_timeout_and_interruption_terminate_only_owned_process_and_keep_cost(self):
        for kind in ('deadline', 'interrupt', 'failure'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                process = Mock(pid=12345, returncode=None)
                process.poll.side_effect = lambda: process.returncode
                calls = []
                def wait(timeout=None):
                    calls.append(timeout)
                    if kind == 'interrupt' and len(calls) == 1:
                        raise KeyboardInterrupt()
                    process.returncode = 7 if kind == 'failure' else -15
                    return process.returncode
                process.wait.side_effect = wait
                record = {'command': ['python', 'probe'], 'resources': {'uuid': 'GPU-probe'}}
                clock = [0., 95., 96.] if kind == 'deadline' else [0., 1., 2.]
                persist = Mock()
                with patch(MODULE + 'subprocess.Popen', return_value=process), patch(MODULE + 'time.monotonic', side_effect=clock), \
                        patch(MODULE + 'gpu_resources', return_value={'uuid': 'GPU-probe', 'memory_total_mib': '40000', 'memory_free_mib': '38000'}), \
                        patch(MODULE + 'os.killpg') as kill:
                    with self.assertRaises((TimeoutError, KeyboardInterrupt, RuntimeError)):
                        execute_probe(record, Path(directory) / 'probe.log', gpu=2, cwd=Path(directory),
                                      environment={}, timeout=100., persist=persist)
                if kind == 'failure':
                    kill.assert_not_called()
                else:
                    kill.assert_called_once_with(12345, signal.SIGTERM)
                self.assertEqual(record['returncode'], 7 if kind == 'failure' else -15)
                self.assertGreater(record['seconds'], 0)
                self.assertEqual(persist.call_count, 2)

    def test_success_records_full_allocation_and_device_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=12345, returncode=0)
            process.wait.return_value = 0
            process.poll.return_value = 0
            record = {'command': ['python', 'probe'], 'resources': {'uuid': 'GPU-probe'}}
            with patch(MODULE + 'subprocess.Popen', return_value=process), \
                    patch(MODULE + 'time.monotonic', side_effect=[0., 1., 2.]), \
                    patch(MODULE + 'gpu_resources', return_value={'uuid': 'GPU-probe', 'memory_total_mib': '40000', 'memory_free_mib': '38000'}), \
                    patch(MODULE + 'os.killpg') as kill:
                execute_probe(record, Path(directory) / 'probe.log', gpu=2, cwd=Path(directory),
                              environment={}, timeout=100., persist=Mock())
            kill.assert_not_called()
            self.assertEqual(record['returncode'], 0)
            self.assertEqual(record['seconds'], 2.)
            self.assertEqual(record['peak_device_memory_used_mib'], 2000.)

    def test_expired_termination_grace_escalates_and_records_all_cleanup_time(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=12345, returncode=None)
            process.poll.return_value = None
            def wait(timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired('probe', timeout)
                process.returncode = -9
                return -9
            process.wait.side_effect = wait
            record = {'command': ['python', 'probe'], 'resources': {'uuid': 'GPU-probe'}}
            with patch(MODULE + 'subprocess.Popen', return_value=process), \
                    patch(MODULE + 'time.monotonic', side_effect=[0., 95., 106.]), \
                    patch(MODULE + 'gpu_resources', return_value={'uuid': 'GPU-probe', 'memory_total_mib': '40000', 'memory_free_mib': '38000'}), \
                    patch(MODULE + 'os.killpg') as kill:
                with self.assertRaises(TimeoutError):
                    execute_probe(record, Path(directory) / 'probe.log', gpu=2, cwd=Path(directory),
                                  environment={}, timeout=100., persist=Mock())
            self.assertEqual([c.args for c in kill.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
            self.assertEqual(record['returncode'], -9)
            self.assertEqual(record['seconds'], 106.)


if __name__ == '__main__':
    unittest.main()
