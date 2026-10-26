"""Independent directions and sampling convergence against analytic test dynamics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_policy_fidelity import fixture
from experiments.ablation.estimator_convergence import (
    VectorMoments, analyze, random_directions, run_convergence, validate_design,
)
from experiments.ablation.policy_estimators import estimate_pair, stochastic_probe
from experiments.ablation.policy_fidelity import parameter_vector
from experiments.ablation.run_estimator_convergence import forecast_seconds


class ConvergenceTests(unittest.TestCase):
    def test_resource_forecast_scales_estimator_and_reference_work_separately(self):
        plan = dict(qualification=dict(counts=[2,4], directions=2, reference_count=4),
                    experiment=dict(counts=[16,64,256], directions=8, reference_count=32),
                    forecast_safety_factor=1.25)
        # 20 s fixed, 4 s gradient work (x64), 16 s paired references (x32).
        forecast = forecast_seconds(dict(seconds=40), dict(reference_phase_seconds=16),
                                    [dict(bptt_seconds=3, ppo_seconds=1)], plan)
        self.assertEqual(forecast, (20 + 4*64 + 16*32)*1.25)

    def test_streamed_full_vector_statistics_match_direct_covariance(self):
        values = torch.randn(17, 31, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        moments = VectorMoments(values[0])
        for vector in values:
            moments.add(vector)
        torch.testing.assert_close(moments.mean, values.mean(0), rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(moments.summary()['variance_trace'], float(values.var(0).sum()), places=12)

    def test_directions_are_fixed_without_using_global_rng_or_gradients(self):
        signs, directions = random_directions(127, 8, 50001)
        torch.manual_seed(19)
        np.random.seed(39)
        signs2, directions2 = random_directions(127, 8, 50001)
        np.testing.assert_array_equal(signs, signs2)
        np.testing.assert_array_equal(directions, directions2)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-14)
        np.testing.assert_array_equal(np.unpackbits(np.packbits(signs, axis=-1), axis=-1)[:, :127], signs)

    def test_end_to_end_prefixes_match_raw_estimates_and_matched_derivatives(self):
        probe = stochastic_probe(fixture(1)[0], .1)
        initial = parameter_vector(probe.policy).clone()
        initial_seed = probe.seed
        calls = []
        def physical():
            calls.append(probe.seed)
            result = probe.objective()
            return dict(proxy=result['proxy'], hard=result['proxy'], initial_state_drift=torch.zeros(probe.batch))
        with tempfile.TemporaryDirectory() as directory:
            result = run_convergence(probe, physical, directory, gradient_seeds=[1,2,3,4],
                reference_seeds=[11,12,13,14], counts=[2,4], epsilons=[1e-4,3e-4],
                directions=3, direction_seed=42, repeats=3)
            self.assertEqual(result['physical_rollouts'], 4 * (1 + 2*3*2) + 2)
            self.assertEqual(result['wm_reference_rollouts'], result['physical_rollouts'])
            self.assertEqual(set(calls), {11,12,13,14})
            self.assertEqual(probe.seed, initial_seed)
            self.assertTrue(torch.equal(parameter_vector(probe.policy), initial))
            raw = dict(np.load(Path(directory)/'raw-reference.npz'))
            np.testing.assert_array_equal(raw['wm'], raw['physical_proxy'])
            directions = random_directions(len(initial),3,42)[1]
            estimates = {m: [] for m in ('bptt','ppo')}
            for seed in (1,2,3,4):
                probe.seed = seed
                pair = estimate_pair(probe)
                for m, method in enumerate(estimates):
                    estimates[method].append(pair[method])
                    np.testing.assert_allclose(raw['projections'][m,seed-1], directions @ pair[method].numpy(), atol=1e-12)
            means = torch.load(Path(directory)/'final-means.pt',weights_only=True)
            for method in estimates:
                torch.testing.assert_close(means[method],torch.stack(estimates[method]).mean(0))
            original = (Path(directory)/'convergence.csv').read_bytes()
            analyze(raw,Path(directory)/'reanalysis')
            self.assertEqual(original,(Path(directory)/'reanalysis/convergence.csv').read_bytes())
            self.assertEqual(json.loads((Path(directory)/'convergence-progress.json').read_text())['status'],'complete')

    def test_failed_physical_probe_restores_parameters_and_retains_failure(self):
        probe = stochastic_probe(fixture(1)[0],.1)
        before = parameter_vector(probe.policy).clone()
        seed = probe.seed
        calls = 0
        def physical():
            nonlocal calls
            calls += 1
            if calls == 4:  # Two baselines, one repeat, then a displaced policy.
                raise RuntimeError('injected physics failure')
            result=probe.objective()
            return dict(proxy=result['proxy'],hard=result['proxy'],initial_state_drift=torch.zeros(probe.batch))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'injected physics'):
                run_convergence(probe,physical,directory,gradient_seeds=[1,2],reference_seeds=[11,12],
                    counts=[2],epsilons=[.001,.003],directions=2,direction_seed=42,repeats=2)
            self.assertTrue(torch.equal(parameter_vector(probe.policy),before))
            self.assertEqual(probe.seed,seed)
            record=json.loads((Path(directory)/'convergence-progress.json').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertFalse((Path(directory)/'convergence-summary.json').exists())

    def test_overlap_invalid_prefixes_and_budget_stop_are_rejected(self):
        for counts,refs,direction in (([2,3],[11,12],42),([2],[2,11],42),([2],[11,12],1)):
            with self.assertRaises(ValueError):
                validate_design([1,2],refs,counts,[.001,.003],2,direction)
        probe=stochastic_probe(fixture(1)[0],.1)
        seed=probe.seed
        def expired():
            raise TimeoutError('injected budget')
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(TimeoutError):
            run_convergence(probe,lambda: None,directory,gradient_seeds=[1,2],reference_seeds=[11,12],
                counts=[2],epsilons=[.001,.003],directions=2,direction_seed=42,check_budget=expired)
        self.assertEqual(probe.seed,seed)


if __name__ == '__main__':
    unittest.main()
