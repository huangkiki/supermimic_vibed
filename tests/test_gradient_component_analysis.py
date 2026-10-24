from copy import deepcopy
import unittest

from test_policy_fidelity import fixture
from experiments.ablation.analyze_gradient_components import component_statistics


def captures():
    result = {}
    for no_critic in (False, True):
        for detach in (False, True):
            probe, optimizer = fixture(8, no_critic=no_critic, detach_action_reward=detach)
            result[no_critic, detach] = probe.capture(optimizer)
    return result


class GradientComponentAnalysisTests(unittest.TestCase):
    def test_original_actor_gradients_obey_additive_objective_identity(self):
        result = component_statistics(captures())
        self.assertLess(result['additive_residual_relative_l2'], 1e-12)
        self.assertAlmostEqual(sum(row['fraction_of_full_projection'] for row in result['components']), 1.)
        self.assertGreater(result['components'][1]['l2'], 0)
        self.assertGreater(result['components'][2]['l2'], 0)

    def test_missing_mismatched_or_nonadditive_capture_is_rejected(self):
        original = captures()
        for kind in ('missing', 'horizon', 'seed', 'label', 'objective', 'gradient'):
            with self.subTest(kind=kind):
                data = deepcopy(original)
                if kind == 'missing':
                    del data[True, True]
                elif kind == 'label':
                    data[True, True]['no_critic'] = False
                elif kind == 'objective':
                    data[True, True]['objective'] += 1
                elif kind == 'gradient':
                    data[True, True]['raw_ascent_gradient'] += 1
                else:
                    data[True, True][kind] += 1
                with self.assertRaises((ValueError, AssertionError)):
                    component_statistics(data)


if __name__ == '__main__':
    unittest.main()
