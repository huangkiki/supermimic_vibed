"""Analytic and source-path checks for the auxiliary stochastic-target comparison."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch
from torch import nn

from test_policy_fidelity import fixture
from experiments.ablation.policy_estimators import (
    FixedGaussianPolicy, collect_score_rollout, estimate_pair, pathwise_gradient,
    score_gradient, stochastic_probe,
)
from experiments.ablation.policy_fidelity import displaced_parameters, parameter_vector
from experiments.ablation.compare_policy_estimators import run_comparison, validate_design


class ConstantMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(.2, dtype=torch.float64))

    def forward(self, history, goal):
        return self.mean.expand(len(history), 1)


class ScalarDynamics(nn.Module):
    def step(self, state, action, xi):
        return tuple(.9 * part + .1 * action for part in state)


class WeightedLinearCritic(nn.Module):
    def __init__(self, quadrature):
        super().__init__()
        self.register_buffer('weights', quadrature*len(quadrature))

    def forward(self, features):
        return .4*features[..., 0]*self.weights


class EstimatorTests(unittest.TestCase):
    def test_common_actions_target_rng_and_model_snapshots_are_preserved(self):
        original, _ = fixture(8)
        before = [deepcopy(module.state_dict()) for module in (original.policy, original.wm, original.critic)]
        rng = torch.random.get_rng_state().clone()
        probe = stochastic_probe(original, .1)
        pair = estimate_pair(probe)
        self.assertFalse(torch.equal(pair['bptt'], pair['ppo']))
        self.assertEqual(pair['imagined_steps_per_estimator'], 16)
        torch.testing.assert_close(pair['rollout']['objective'], probe.objective()['proxy'], rtol=0, atol=0)
        torch.testing.assert_close(pair['rollout']['action'], probe.objective()['actions'], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for module, snapshot in zip((original.policy, original.wm, original.critic), before):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        self.assertIsNone(probe.policy.records)
        self.assertTrue(all(p.grad is None for p in original.wm.parameters()))

    def test_pathwise_gradient_matches_common_noise_finite_difference(self):
        for horizon in (1, 8, 32):
            with self.subTest(horizon=horizon):
                original, _ = fixture(horizon)
                probe = stochastic_probe(original, .1)
                gradient = pathwise_gradient(probe)['gradient']
                direction = gradient / gradient.norm()
                with displaced_parameters(probe.policy, 1e-5 * direction):
                    plus = probe.objective()['proxy'].mean()
                with displaced_parameters(probe.policy, -1e-5 * direction):
                    minus = probe.objective()['proxy'].mean()
                torch.testing.assert_close((plus-minus)/2e-5, gradient @ direction, rtol=1e-6, atol=1e-8)

    def test_ppo_is_detached_from_policy_dependent_rollout_storage(self):
        original, _ = fixture(8)
        probe = stochastic_probe(original, .1)
        rollout = collect_score_rollout(probe)
        expected = score_gradient(probe.policy, rollout)['gradient']
        parameter = next(probe.policy.parameters()).sum()
        # Identical values, but a nonzero path to policy parameters if not detached.
        attached = {key: value + 100 * (parameter-parameter.detach())
                    if torch.is_tensor(value) else value for key, value in rollout.items()}
        actual = score_gradient(probe.policy, attached)['gradient']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with displaced_parameters(probe.policy, .1 * torch.ones_like(parameter_vector(probe.policy))):
            with self.assertRaisesRegex(ValueError, 'theta_old'):
                score_gradient(probe.policy, rollout)

    def test_both_estimators_integrate_to_analytic_gaussian_gradient(self):
        # Exact product Gauss-Hermite integration, not Monte Carlo and not another
        # implementation of the same backprop formula. H8 checks causal tails,
        # discounted survival weights and the H scaling of the reused PPO loss.
        nodes, weights = np.polynomial.hermite.hermgauss(3)
        for horizon in (1, 8):
            with self.subTest(horizon=horizon):
                indices = np.array(list(itertools.product(range(3), repeat=horizon)))
                batch = len(indices)
                quadrature = torch.tensor(weights[indices].prod(1) / np.pi**(horizon/2), dtype=torch.float64)
                noise = torch.tensor(np.sqrt(2) * nodes[indices].T, dtype=torch.float64).unsqueeze(-1)
                original, _ = fixture(horizon, no_critic=True)
                original.policy = ConstantMean()
                original.wm = ScalarDynamics()
                original.batch = batch
                original.starts = {key: value[:1].expand(batch, *value.shape[1:]).clone()
                                   for key, value in original.starts.items()}
                original.history = {key: value[:1].expand(batch, *value.shape[1:]).clone()
                                    for key, value in original.history.items()}
                original.starts['a_prev'] = original.starts['a_prev'][:, :1]
                original.history['a_prev'] = original.history['a_prev'][:, :, :1]
                original.mdp.add_noise = lambda obs: obs
                original.mdp.soft_termination = lambda state, frames: torch.full_like(state[0][..., 0], .1)
                original.mdp.reward = lambda state, frames, action, previous: (
                    -(action[:, 0]-.7).square() * quadrature.repeat(len(action)//batch) * batch)
                if horizon == 1:
                    original.no_critic = False
                    original.critic = WeightedLinearCritic(quadrature)
                probe = stochastic_probe(original, .15)
                def draw_like(value):
                    nonlocal draws
                    result = noise[draws]
                    draws += 1
                    self.assertEqual(result.shape, value.shape)
                    return result
                draws = 0
                with patch('torch.randn_like', side_effect=draw_like):
                    bptt = pathwise_gradient(probe)['gradient']
                draws = 0
                with patch('torch.randn_like', side_effect=draw_like):
                    rollout = collect_score_rollout(probe)
                ppo = score_gradient(probe.policy, rollout)['gradient']
                expected = torch.tensor([sum((.99*.9)**t for t in range(horizon))], dtype=torch.float64)
                if horizon == 1:
                    expected += .99*.9*.4*.1
                torch.testing.assert_close(bptt, expected, rtol=2e-7, atol=1e-8)
                torch.testing.assert_close(ppo, expected, rtol=2e-7, atol=1e-8)
                self.assertEqual(draws, horizon)

    def test_end_to_end_comparison_retains_independent_noise_and_equal_step_budgets(self):
        probe = stochastic_probe(fixture(8)[0], .1)
        calls = []
        def physical():
            calls.append(probe.seed)
            result = probe.objective()
            return dict(proxy=result['proxy'], hard=result['proxy'], initial_state_drift=torch.zeros(probe.batch))
        with tempfile.TemporaryDirectory() as directory:
            summary = run_comparison(probe, physical, directory, gradient_seeds=[1,2],
                                     reference_seeds=[101,102,103], epsilons=[.001,.003])
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(len(calls), 3*(1+4*2)+2)
            self.assertEqual(set(calls), {101,102,103})
            self.assertEqual(probe.seed, 9001)
            saved = torch.load(Path(directory)/'mean-gradients.pt', weights_only=False)
            for method in ('bptt','ppo'):
                draws = torch.stack([torch.load(Path(directory)/f'estimate-{seed}.pt', weights_only=False)[method]
                                     for seed in (1,2)]).double()
                torch.testing.assert_close(saved['means'][method], draws.mean(0), rtol=0, atol=0)
            result = torch.load(Path(directory)/'physical-reference.pt', weights_only=False)
            self.assertEqual(len(result['rows']), 8)
            self.assertTrue(all(abs(row['positive_displacement_l2']-row['epsilon'])<1e-12 for row in result['rows']))
            self.assertTrue(all(row['numerical_repeat_noise']==0 for row in result['rows']))
            receipt=json.loads((Path(directory)/'comparison-summary.json').read_text())
            self.assertIn('noise tapes', receipt['independent_unit'])

    def test_disjoint_noise_validation_and_budget_failure_preserve_seed(self):
        for gradients, references in (([1,1],[2,3]),([1,2],[2,3]),([1],[2,3])):
            with self.assertRaises(ValueError):
                validate_design(gradients,references,[.001,.003])
        probe = stochastic_probe(fixture(1)[0], .1)
        def expired():
            raise TimeoutError('budget')
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TimeoutError):
            run_comparison(probe, lambda: None, directory, gradient_seeds=[1,2], reference_seeds=[3,4],
                           epsilons=[.001,.003],check_budget=expired)
        self.assertEqual(probe.seed,9001)

    def test_invalid_distribution_and_surrogate_target_are_rejected(self):
        original, _ = fixture(1)
        for std in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                stochastic_probe(original, std)
        original.detach_action_reward = True
        with self.assertRaises(ValueError):
            stochastic_probe(original, .1)
        with self.assertRaises(ValueError):
            collect_score_rollout(fixture(1)[0])


if __name__ == '__main__':
    unittest.main()
