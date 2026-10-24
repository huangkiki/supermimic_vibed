import csv
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.ablation.compare_fidelity import compare
from experiments.ablation.fidelity import collect_bank


class FakeEnvironment:
    """History-sensitive collector test without native physics dependencies."""

    def __init__(self, fail_first=False):
        self.N, self.device = 2, 'cpu'
        self.motion = SimpleNamespace(
            names=['walk', 'run'], clip_len=torch.tensor([100, 100]),
            clip_start=torch.tensor([0, 100]), q=torch.full((200, 1), 1000.),
            goal=lambda frames, count: frames[:, None])
        self.mdp = SimpleNamespace(obs=lambda state, action, xi, frames: frames.float()[:, None])
        self.default_q, self.alpha = torch.zeros(1), torch.ones(1)
        self.f = torch.zeros(2, dtype=torch.long)
        self.a_prev, self.xi = torch.zeros(2, 1), torch.zeros(2, 33)
        self.done_eval = torch.zeros(2, dtype=torch.bool)
        self.failed_eval = torch.zeros(2, dtype=torch.bool)
        self.resets, self.fail_first = 0, fail_first

    def reset(self, ids, frames):
        self.resets += 1
        self.f.copy_(frames)
        self.a_prev.zero_()

    def refresh(self):
        pass

    def state(self):
        return self.f.float()[:, None]

    def step(self, action):
        self.f += 1
        self.a_prev.copy_(action)
        if self.fail_first and self.resets == 1:
            self.failed_eval.fill_(True)


class BankTests(unittest.TestCase):
    def collect(self, controller, fail_first=False):
        env = FakeEnvironment(fail_first)
        mdp = ModuleType('supermimic.mdp')
        mdp.flatten_state = lambda state: state.clone()
        cfg = SimpleNamespace(seed=0, hist=3, goal_frames=1)
        policy = lambda history, goal: history.mean(1)
        with patch.dict(sys.modules, {'supermimic.mdp': mdp}), patch(
            'experiments.ablation.fidelity.contact_signature', return_value=[(), ()]
        ):
            bank = collect_bank(env, policy, cfg, 2, 2, 3, 9001, controller)
        return bank

    def test_policy_roll_in_keeps_history_across_probe_boundary(self):
        bank = self.collect('policy')
        start = bank['reset_frames'].float()[:, None]
        # With history length 3, first probe action uses f1,f2,f3, not f3,f3,f3.
        torch.testing.assert_close(bank['actions'][0], start + 2)
        torch.testing.assert_close(bank['actions'][1], start + 3)
        torch.testing.assert_close(bank['previous_action'], bank['prefix_actions'][-1])
        torch.testing.assert_close(bank['history_before_probe'][:, :, 0],
                                   start + torch.arange(3))

    def test_reference_control_retains_switch_and_history_initialization(self):
        bank = self.collect('reference_pd')
        torch.testing.assert_close(bank['previous_action'], torch.full((2, 1), 1000.))
        torch.testing.assert_close(bank['actions'][0], bank['frames'].float()[:, None])
        self.assertIsNone(bank['history_before_probe'])

    def test_rejected_roll_in_resets_policy_history_and_records_rejection(self):
        bank = self.collect('policy', fail_first=True)
        self.assertEqual(bank['sampling_rounds'], 2)
        torch.testing.assert_close(bank['start_sampling_attempts'], torch.full((2,), 2))
        torch.testing.assert_close(bank['actions'][0], bank['reset_frames'].float()[:, None] + 2)


class PairedBankIdentityTests(unittest.TestCase):
    def write_probe(self, root, objective, bank_sha, bank_path='/shared/bank.pt', prefix='policy'):
        root.mkdir()
        receipt = dict(collector_sha256='collector', checkpoint_sha256={'ckpt_000100.pt': 'model'},
                       gpu='A100', precision='fp32', reset_protocol='fixed',
                       manifest={'gpu_uuid': 'GPU-one'},
                       args=dict(bank=bank_path, gradient_horizon=1, samples=1, directions=1,
                                 epsilons=[.01, .03], contact_level='body', probe_seed=9001,
                                 prefix_steps=16, prefix_controller=prefix, reward_objective=objective))
        if bank_sha is not None:
            receipt['bank_sha256'] = bank_sha
        (root / 'probe.json').write_text(json.dumps(receipt))
        row = dict(checkpoint_sha256='model', objective='soft', epsilon=.01, sample=0, direction=0,
                   direction_type='random_unit_l2', contact_group='stable_contact', resolved=True,
                   predicted_derivative=1., fd_derivative=1.)
        with (root / 'gradient.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_same_path_with_different_bank_content_cannot_be_paired(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'state'
            self.write_probe(a, 'full', 'a' * 64)
            self.write_probe(b, 'state_only', 'b' * 64)
            with self.assertRaisesRegex(ValueError, 'bank'):
                compare(a, b)

    def test_identical_bank_copy_can_move_without_changing_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'state'
            self.write_probe(a, 'full', 'a' * 64, '/machine-one/bank.pt')
            self.write_probe(b, 'state_only', 'a' * 64, '/machine-two/bank.pt')
            rows = compare(a, b)
            self.assertEqual(rows[0]['bank_identity'], 'sha256')

    def test_two_unidentified_new_banks_are_not_a_paired_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'state'
            self.write_probe(a, 'full', None, None)
            self.write_probe(b, 'state_only', None, None)
            with self.assertRaisesRegex(ValueError, 'bank'):
                compare(a, b)

    def test_legacy_path_only_evidence_is_labeled_and_cannot_mix_with_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'state'
            self.write_probe(a, 'full', None, prefix='reference_pd')
            self.write_probe(b, 'state_only', None, prefix='reference_pd')
            self.assertEqual(compare(a, b)[0]['bank_identity'], 'legacy_path_only')
            receipt = json.loads((a / 'probe.json').read_text())
            receipt['bank_sha256'] = 'a' * 64
            (a / 'probe.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'bank'):
                compare(a, b)

    def test_paired_metadata_cannot_hide_a_changed_roll_in_or_physical_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'state'
            self.write_probe(a, 'full', 'a' * 64)
            self.write_probe(b, 'state_only', 'a' * 64)
            receipt = json.loads((b / 'probe.json').read_text())
            receipt['args']['prefix_controller'] = 'reference_pd'
            (b / 'probe.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'prefix'):
                compare(a, b)
            receipt['args']['prefix_controller'] = 'policy'
            receipt['manifest']['gpu_uuid'] = 'GPU-two'
            (b / 'probe.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'GPU UUID'):
                compare(a, b)


if __name__ == '__main__':
    unittest.main()
