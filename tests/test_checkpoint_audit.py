from copy import copy, deepcopy
import json
import unittest

import torch
from torch import nn

from test_policy_fidelity import fixture, TinyWM
from experiments.ablation.checkpoint_audit import bias_slice, repeat_bank, validate_plan
from experiments.ablation.policy_estimators import collect_score_rollout, score_gradient, stochastic_probe
from experiments.ablation.run_checkpoint_audit import forecast_seconds
from experiments.ablation.policy_fidelity_run import parser


class CheckpointAuditTests(unittest.TestCase):
    def test_native_audit_arguments_are_json_serializable_before_initialization(self):
        args = parser().parse_args(['--checkpoint', 'checkpoint.pt', '--bank', 'bank.pt', '--output', 'out',
                                   '--gpu', '2', '--gpu-uuid', 'gpu', '--checkpoint-audit', 'plan.json'])
        self.assertEqual(json.loads(json.dumps(vars(args)))['checkpoint_audit'], 'plan.json')

    def test_forecast_scales_reference_axes_and_preserves_setup_cost(self):
        small = dict(gradient_seeds=[1, 2], reference_seeds=[3, 4], directions=2)
        full = dict(gradient_seeds=list(range(16)), reference_seeds=list(range(16)), directions=8)
        timings = [dict(operation='ppo_sim', seconds=2),
                   dict(operation='physical_independent_direction', seconds=3),
                   dict(operation='physical_repeat', seconds=1)]
        self.assertEqual(forecast_seconds(dict(seconds=10), timings, small, full, 1.3),
                         1.3 * (4*8 + 2*8 + 3*32 + 1))

    def test_real_transition_score_uses_supplied_dynamics_and_is_detached(self):
        original, _ = fixture(8)
        probe = stochastic_probe(original, .1)
        replacement = TinyWM()
        with torch.no_grad():
            replacement.gain.fill_(.21)
        replacement_probe = copy(probe)
        replacement_probe.wm = replacement
        actual = collect_score_rollout(probe, dynamics=replacement)
        expected = collect_score_rollout(replacement_probe)
        ordinary = collect_score_rollout(probe)
        torch.testing.assert_close(actual['objective'], expected['objective'], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual['objective'], ordinary['objective']))
        a = score_gradient(probe.policy, actual)['gradient']
        b = score_gradient(probe.policy, expected)['gradient']
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(not x.requires_grad for x in actual['states']))
        self.assertTrue(all(p.grad is None for p in replacement.parameters()))
        self.assertEqual(actual['states'][0].shape[:2], (9, 2))

    def test_replication_preserves_every_state_history_and_contact_axis(self):
        original, _ = fixture(8)
        bank = dict(state=original.starts['s'], frames=original.starts['f'], xi=original.starts['xi'],
                    previous_action=original.starts['a_prev'], replay_history=original.history,
                    actions=torch.arange(12).view(3, 2, 2),
                    prefix_contacts=[[((0, 1),), ()]], motion_names=['a', 'b'], source_sample_indices=[3, 7])
        frozen = deepcopy(bank)
        actual = repeat_bank(bank, 3)
        self.assertEqual(actual['frames'].shape[0], 6)
        for j in range(3):
            torch.testing.assert_close(actual['state'][2*j:2*j+2], bank['state'], rtol=0, atol=0)
            for key in bank['replay_history']:
                torch.testing.assert_close(actual['replay_history'][key][2*j:2*j+2], bank['replay_history'][key])
        self.assertEqual(actual['prefix_contacts'][0], bank['prefix_contacts'][0]*3)
        torch.testing.assert_close(bank['state'], frozen['state'], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            repeat_bank(bank, 0)

    def test_bias_selection_follows_parameter_identity(self):
        mean = nn.Module()
        mean.enc = nn.Linear(3, 2)
        mean.dec = nn.Sequential(nn.Linear(2, 4))
        mean.register_parameter('after', nn.Parameter(torch.ones(5)))
        actor = nn.Module()
        actor.mean_policy = mean
        section = bias_slice(actor)
        vector = torch.cat([p.detach().flatten() for p in actor.parameters()])
        torch.testing.assert_close(vector[section], mean.dec[-1].bias)
        self.assertEqual(section.stop-section.start, 4)

    def test_reference_leakage_and_undeclared_primary_scale_are_rejected(self):
        plan = dict(gradient_seeds=[1, 2], reference_seeds=[3, 4], epsilons=[.001, .003, .01],
                    directions=2, direction_seed=5, wm_checkpoints=['a', 'b'], primary_epsilon=.003,
                    replicas=16, repeats=3)
        validate_plan(plan)
        for field, value in [('reference_seeds', [2, 3]), ('primary_epsilon', .002), ('wm_checkpoints', ['a', 'a'])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_plan({**plan, field: value})


if __name__ == '__main__':
    unittest.main()
