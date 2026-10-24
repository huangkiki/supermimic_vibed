"""CPU checks prove source-path parity, not fidelity of a learned robot model."""
import ast
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from experiments.ablation.policy_fidelity import (
    PolicyGradientProbe, TRAIN_SOURCE, displaced_parameters, fixed_rng,
    parameter_vector, training_actor_source,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.17, -.08, .04], [-.03, .12, .09]], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([.05, -.02], dtype=torch.float64))

    def forward(self, history, goal):
        inputs = torch.cat((history.mean(1), goal.to(history.dtype)), -1)
        return torch.tanh(inputs @ self.weight.T + self.bias)


class TinyWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.07, dtype=torch.float64))

    def step(self, state, action, xi):
        # Smooth stable recurrence with feedback through both observation and action.
        return tuple(.94 * part + self.gain * action[:, :1] + .015 * action[:, 1:]
                     for part in state)


class TinyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('updates', torch.tensor(0))

    def update(self, features):
        self.updates += 1


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.norm = TinyNorm()

    def forward(self, features):
        return self.scale * features[..., 0] - .1 * features[..., 0].square()


class TinyMDP:
    def obs(self, state, previous, xi, frames):
        return torch.cat((state[0], previous[..., :1]), -1)

    def add_noise(self, obs):
        return obs + .003 * torch.randn_like(obs)

    def reward(self, state, frames, action, previous):
        return -.7 * (state[0][..., 0] - .4).square() - .2 * (action - previous).square().sum(-1)

    def soft_termination(self, state, frames):
        return .03 * torch.sigmoid(3 * state[0][..., 0])

    def critic_features(self, state, frames, xi):
        return state[0]


def fixture(horizon=32, **kwargs):
    policy, wm, critic = TinyPolicy(), TinyWM(), TinyCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.02, betas=(.9, .95))
    # Deliberately nonzero, anisotropic moments: fresh Adam must differ.
    for scale in (.3, -.17, .7):
        optimizer.zero_grad()
        loss = sum(((p - scale) ** 2).sum() for p in policy.parameters())
        loss.backward()
        optimizer.step()
    state = torch.tensor([[.1] * 6, [.24] * 6], dtype=torch.float64)
    starts = dict(s=state, a_prev=torch.zeros(2, 2, dtype=torch.float64),
                  xi=torch.ones(2, 1, dtype=torch.float64), f=torch.tensor([2, 5]))
    history = {k: v[:, None].expand(-1, 3, *v.shape[1:]).clone() for k, v in starts.items()}
    # Distinct raw history matters to feedback; newest entry remains the start.
    history['s'][:, 0] -= .03
    motion = SimpleNamespace(goal=lambda f, k: .01 * f[:, None],
                             clip_end=torch.tensor([100]), clip_id=torch.zeros(100, dtype=torch.long))
    probe = PolicyGradientProbe(policy=policy, wm=wm, critic=critic, mdp=TinyMDP(), motion=motion,
        starts=starts, history=history, unflatten_state=lambda x: x.split(1, -1),
        normalize_xi=lambda x: x, horizon=horizon, goal_frames=1, **kwargs)
    return probe, optimizer.state_dict()


class PolicyFidelityTests(unittest.TestCase):
    def test_original_full_function_matches_captured_actor_for_all_horizons(self):
        for horizon in (1, 8, 32):
            with self.subTest(horizon=horizon):
                probe, state = fixture(horizon)
                capture = probe.capture(state)
                full = deepcopy(probe)
                optimizer = torch.optim.Adam(full.policy.parameters())
                optimizer.load_state_dict(deepcopy(state))
                before = parameter_vector(full.policy)
                namespace = full._namespace(full.policy, full.wm, optimizer)
                namespace['args'].lam = .95
                namespace['args'].critic_epochs = 0
                namespace['args'].critic_minibatches = 1
                namespace['critic_input_c'] = namespace['_critic_input']
                # Execute the entire actual source function, including post-step critic norm.
                node = next(n for n in ast.parse(TRAIN_SOURCE.read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == 'policy_update')
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN_SOURCE), 'exec'), namespace)
                with fixed_rng(full.seed, full.device):
                    full_result = namespace['policy_update']()
                torch.testing.assert_close(full_result[0], capture['loss'], rtol=0, atol=0)
                torch.testing.assert_close(parameter_vector(full.policy) - before,
                                           capture['adam_displacement'], rtol=0, atol=0)
                self.assertEqual(int(full.critic.norm.updates), 1)
                self.assertEqual(int(probe.critic.norm.updates), 0)

    def test_closed_loop_directional_derivative_matches_smooth_cpu_reference(self):
        for horizon in (1, 8, 32):
            probe, state = fixture(horizon)
            capture = probe.capture(state)
            gradient = capture['raw_ascent_gradient']
            direction = gradient / gradient.norm()
            epsilon = 1e-5
            with displaced_parameters(probe.policy, epsilon * direction):
                plus = probe.objective()['proxy'].mean()
            with displaced_parameters(probe.policy, -epsilon * direction):
                minus = probe.objective()['proxy'].mean()
            torch.testing.assert_close((plus - minus) / (2 * epsilon), gradient @ direction,
                                       rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(capture['objective'], probe.objective()['proxy'])

    def test_both_clips_and_historical_adam_are_saved(self):
        probe, state = fixture()
        probe.mdp.reward = lambda S, f, a, previous: -1000 * (S[0][:, 0] - 3).square()
        capture = probe.capture(state)
        raw = capture['raw_loss_gradient']
        norm_clip = raw * (1 / (raw.norm() + 1e-6)).clamp(max=1)
        torch.testing.assert_close(capture['norm_clipped_loss_gradient'], norm_clip)
        torch.testing.assert_close(capture['clipped_loss_gradient'], norm_clip.clamp(-.1, .1))
        self.assertGreater(float(raw.norm()), 1)
        fresh_policy = deepcopy(probe.policy)
        fresh = torch.optim.Adam(fresh_policy.parameters(), lr=.02, betas=(.9, .95))
        offset = 0
        for p in fresh_policy.parameters():
            p.grad = capture['clipped_loss_gradient'][offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        before = parameter_vector(fresh_policy)
        fresh.step()
        self.assertFalse(torch.allclose(parameter_vector(fresh_policy) - before, capture['adam_displacement']))
        for old, new in zip(state['state'].values(), capture['optimizer_after']['state'].values()):
            self.assertEqual(float(new['step']), float(old['step']) + 1)

    def test_forward_reward_unchanged_by_direct_action_gradient_ablation(self):
        original, state = fixture()
        detached, _ = fixture(detach_action_reward=True)
        a, b = original.capture(state), detached.capture(state)
        torch.testing.assert_close(a['objective'], b['objective'], rtol=0, atol=0)
        self.assertGreater(float((a['raw_ascent_gradient'] - b['raw_ascent_gradient']).norm()), 1e-5)
        no_critic, _ = fixture(no_critic=True)
        c = no_critic.capture(state)
        self.assertFalse(torch.equal(a['objective'], c['objective']))

    def test_rng_parameters_optimizer_and_normalizers_are_unchanged(self):
        probe, state = fixture()
        probe.wm.gain.requires_grad_(False)
        before = [deepcopy(m.state_dict()) for m in (probe.policy, probe.wm, probe.critic)]
        rng = torch.random.get_rng_state().clone()
        optimizer_before = deepcopy(state)
        first, second = probe.capture(state), probe.capture(state)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(probe.wm.gain.requires_grad)
        self.assertTrue(probe.critic.scale.requires_grad)
        torch.testing.assert_close(first['adam_displacement'], second['adam_displacement'], rtol=0, atol=0)
        for module, saved in zip((probe.policy, probe.wm, probe.critic), before):
            for name, tensor in module.state_dict().items():
                self.assertTrue(torch.equal(tensor, saved[name]))
        for key, value in state['state'].items():
            for name, tensor in value.items():
                self.assertTrue(torch.equal(tensor, optimizer_before['state'][key][name]))

    def test_failure_restores_flags_rng_and_policy(self):
        probe, _ = fixture()
        before, rng = parameter_vector(probe.policy), torch.random.get_rng_state()
        probe.mdp.reward = lambda *args: (_ for _ in ()).throw(RuntimeError('test failure'))
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with displaced_parameters(probe.policy, torch.ones_like(before)):
                probe.objective()
        torch.testing.assert_close(parameter_vector(probe.policy), before, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(probe.wm.gain.requires_grad)
        self.assertTrue(probe.critic.scale.requires_grad)

    def test_missing_adam_history_is_rejected(self):
        probe, state = fixture()
        state['state'].clear()
        with self.assertRaisesRegex(ValueError, 'Historical Adam state is missing'):
            probe.capture(state)

    def test_raw_history_and_clip_boundary_are_required(self):
        probe, _ = fixture()
        probe.history['s'][:, -1] += 1
        arguments = dict(policy=probe.policy, wm=probe.wm, critic=probe.critic, mdp=probe.mdp,
            motion=probe.motion, starts=probe.starts, history=probe.history,
            unflatten_state=probe.unflatten_state, normalize_xi=probe.normalize_xi)
        with self.assertRaisesRegex(ValueError, 'history must end'):
            PolicyGradientProbe(**arguments)
        arguments['history']['s'][:, -1] = probe.starts['s']
        arguments['motion'].clip_end[:] = 8
        with self.assertRaisesRegex(ValueError, 'clip boundary'):
            PolicyGradientProbe(**arguments)

    def test_source_has_no_launch_or_critic_update_and_is_fingerprinted(self):
        _, provenance = training_actor_source()
        self.assertEqual(len(provenance['train_source_sha256']), 64)
        self.assertEqual(provenance['source'], 'supermimic/train.py::policy_update')
        self.assertNotIn('isaaclab', __import__('sys').modules)


class DirectionScanTests(unittest.TestCase):
    def test_paired_scan_and_actual_adam_restore_policy(self):
        from experiments.ablation.policy_fidelity import scan_parameter_directions
        probe, state = fixture(8)
        before = parameter_vector(probe.policy)
        capture = probe.capture(state)
        calls = []

        def smooth_reference():
            calls.append(1)
            result = probe.objective()
            # Distinct synthetic hard score verifies it is not substituted for proxy.
            return dict(proxy=result['proxy'], hard=-result['proxy'])

        result = scan_parameter_directions(probe, capture, smooth_reference,
            epsilons=[1e-4, 3e-4, 1e-3], random_directions=2, repeats=3)
        self.assertEqual(len(calls), 3 + 5 * 3 * 2 + 1)
        self.assertEqual(len(result['rows']), 5 * 3 * 2)
        from experiments.ablation.policy_fidelity import summarize_direction_scan
        summary = summarize_direction_scan(result)
        self.assertEqual(len(summary['summaries']), 2 * 3 * 2)
        for row in result['rows']:
            self.assertEqual(row['baseline_repeat_noise'], 0)
            if row['objective'] == 'proxy':
                self.assertTrue(row['sign_agrees'])
                self.assertTrue(row['resolved'])
                self.assertAlmostEqual(row['predicted_derivative'], row['numerical_derivative'], places=5)
            else:
                self.assertFalse(row['sign_agrees'])
                self.assertEqual(row['interpretation'], 'update_direction_effectiveness_only')
        torch.testing.assert_close(parameter_vector(probe.policy), before, rtol=0, atol=0)
        self.assertAlmostEqual(result['actual_adam_improvement']['hard'],
                               -result['actual_adam_improvement']['proxy'])

    def test_direct_reward_ablation_is_not_labeled_matching_derivative(self):
        from experiments.ablation.policy_fidelity import scan_parameter_directions
        probe, state = fixture(1, detach_action_reward=True)
        def reference():
            value = probe.objective()['proxy']
            return dict(proxy=value, hard=value)
        result = scan_parameter_directions(probe, probe.capture(state), reference,
            epsilons=[1e-4, 1e-3], random_directions=1, repeats=2)
        self.assertTrue(all(row['interpretation'] == 'update_direction_effectiveness_only' for row in result['rows']))


class NativeAdapterContractTests(unittest.TestCase):
    """Use a deterministic CPU environment double; this is NOT a MuJoCo run."""
    def test_closed_loop_physics_uses_source_obs_noise_and_masks_only_hard_return(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from experiments.ablation.policy_fidelity import ClosedLoopPhysical
        probe, _ = fixture(8)
        bank = dict(state=probe.starts['s'].clone(), frames=probe.starts['f'].clone(),
                    xi=probe.starts['xi'].clone(), previous_action=probe.starts['a_prev'].clone())

        class Environment:
            eval_mode = True
            def state(self):
                return self.value.split(1, -1)
            def step(self, action):
                self.value = torch.cat(probe.wm.step(self.state(), action, None), -1)
                self.count += 1
                return {'done': torch.full((probe.batch,), self.count >= 3)}
        env = Environment()
        def restore(env, bank):
            env.value, env.count = bank['state'].clone(), 0
        mdp_module = ModuleType('supermimic.mdp')
        mdp_module.flatten_state = lambda S: torch.cat(S, -1)
        with patch.dict(sys.modules, {'supermimic.mdp': mdp_module}), patch('experiments.ablation.fidelity.restore_probe_state', restore):
            physical = ClosedLoopPhysical(probe, env, bank)
            result = physical()
            repeat = physical()
        expected = probe.objective()
        torch.testing.assert_close(result['proxy'], expected['proxy'], rtol=0, atol=0)
        torch.testing.assert_close(result['actions'], expected['actions'], rtol=0, atol=0)
        expected_hard = sum(probe.gamma ** i * result['rewards'][i] for i in range(3))
        torch.testing.assert_close(result['hard'], expected_hard)
        torch.testing.assert_close(result['hard'], repeat['hard'], rtol=0, atol=0)
        self.assertEqual(len(result['actions']), 8)
        self.assertEqual(env.count, 8)

    def test_bank_selection_and_raw_history_reconstruction(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from experiments.ablation.policy_fidelity_run import reconstruct_history, select_bank
        from experiments.ablation.policy_fidelity import validate_starts
        state = torch.zeros(3, 6)
        prefix = torch.tensor([[[1.], [2.], [3.]], [[4.], [5.], [6.]]])
        bank = dict(state=state + torch.tensor([[5.], [7.], [9.]]), frames=torch.tensor([2, 2, 2]),
                    reset_frames=torch.zeros(3, dtype=torch.long), xi=torch.ones(3, 1),
                    previous_action=prefix[-1], prefix_actions=prefix,
                    history_before_probe=None, motion_names=['a', 'b', 'c'])
        selected = select_bank(bank, [2, 0])
        self.assertEqual(selected['motion_names'], ['c', 'a'])
        self.assertEqual(len(bank['frames']), 3)
        class Environment:
            N, device = 2, 'cpu'
            xi = torch.ones(2, 1)
            failed_eval = torch.zeros(2, dtype=torch.bool)
            done_eval = torch.zeros(2, dtype=torch.bool)
            def reset(self, ids, frames):
                self.value, self.f, self.a_prev = torch.zeros(2, 6), frames.clone(), torch.zeros(2, 1)
            def refresh(self):
                pass
            def state(self):
                return self.value
            def step(self, action):
                self.value += action
                self.a_prev = action.clone()
                self.f += 1
        mdp_module = ModuleType('supermimic.mdp')
        mdp_module.flatten_state = lambda S: S.clone()
        with patch.dict(sys.modules, {'supermimic.mdp': mdp_module}):
            history, provenance = reconstruct_history(Environment(), selected, 4)
        starts = dict(s=selected['state'], f=selected['frames'], xi=selected['xi'], a_prev=selected['previous_action'])
        validate_starts(starts, history, 2, 4)
        torch.testing.assert_close(history['f'], torch.tensor([[0, 0, 1, 2], [0, 0, 1, 2]]))
        self.assertFalse(provenance['historical_training_update_recovered'])

    def test_entry_dispatch_preserves_action_default(self):
        from unittest.mock import patch
        from experiments.ablation.fidelity import dispatch
        with patch('experiments.ablation.fidelity.main') as action, patch('experiments.ablation.policy_fidelity_run.main') as policy:
            dispatch(['--checkpoints', 'old.pt'])
            action.assert_called_once_with(['--checkpoints', 'old.pt'])
            dispatch(['--probe-space', 'policy', '--checkpoint', 'new.pt'])
            policy.assert_called_once_with(['--checkpoint', 'new.pt'])

    def test_busy_gpu_is_rejected_before_loading_checkpoint(self):
        from unittest.mock import patch
        from experiments.ablation.policy_fidelity_run import main
        with patch('experiments.ablation.run_suite.wait_for_idle_gpu', side_effect=RuntimeError('busy')), patch('torch.load') as load:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                main(['--checkpoint', 'missing.pt', '--bank', 'missing-bank.pt', '--output', 'unused',
                      '--gpu', '0', '--gpu-uuid', 'GPU-test'])
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
