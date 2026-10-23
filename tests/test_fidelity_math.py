import unittest
import numpy as np
from experiments.ablation.fidelity_math import agreement, contact_groups, reference_quality, window_contact_groups


class FidelityTests(unittest.TestCase):
    def test_noisy_or_scale_dependent_reference_is_not_truth(self):
        self.assertFalse(reference_quality([1, -4, 20], [0.1, 0.2, 0.4], 0).any())
        self.assertFalse(reference_quality([1, 1, 1], [1e-8, 1e-8, 1e-8], 1e-5).any())
        np.testing.assert_array_equal(reference_quality([1, 1.01, 4], [.1, .2, .4], 0), [True, True, False])

    def test_prediction_error_does_not_imply_gradient_accuracy(self):
        # f(x)=x; surrogate x+eps*sin(x/eps^2) is uniformly eps-accurate,
        # yet its derivative at zero is 1+1/eps. No accuracy threshold repairs this.
        eps = 1e-3
        x = np.linspace(-1, 1, 1000)
        error = np.mean((eps * np.sin(x / eps**2))**2)
        self.assertLess(error, 1e-6)
        self.assertGreater(1 + 1 / eps, 100)

    def test_contact_changes_have_a_neighborhood(self):
        data = [[("foot",)], [("foot",)], [()], [()], [()]]
        labels = contact_groups(data, radius=1)
        self.assertEqual([r[0] for r in labels],
                         ["stable_contact", "switch_nearby", "switch_nearby", "switch_nearby", "flight"])

    def test_unresolved_rows_are_reported_not_scored_as_correct(self):
        score = agreement([1., 0., -1.], [2., 10., 1.], [True, False, True])
        self.assertEqual(score["resolved"], 2)
        self.assertEqual(score["sign_agreement"], .5)
        self.assertIsNone(score["pearson"])

    def test_future_switch_does_not_relabel_a_short_stable_window(self):
        labels = [["stable_contact"], ["stable_contact"], ["switch_nearby"]]
        self.assertEqual(window_contact_groups(labels, 0, 2), ["stable_contact"])
        self.assertEqual(window_contact_groups(labels, 0, 3), ["switch_nearby"])
        with self.assertRaises(ValueError):
            window_contact_groups(labels, 0, 4)

    def test_state_only_objective_cannot_inherit_direct_action_gradient(self):
        from types import SimpleNamespace
        import torch
        from experiments.ablation.fidelity import select_probe_reward
        state = torch.tensor(2., requires_grad=True)
        action = torch.tensor(3., requires_grad=True)
        terms = dict(track=state.square(), joint_limit=state.relu())
        config = SimpleNamespace(w_joint_limit=-2.)
        full = terms["track"] - 10. * action.square() - .04 * terms["joint_limit"]
        self.assertIs(select_probe_reward(full, terms, config, "full"), full)
        state_only = select_probe_reward(full, terms, config, "state_only")
        state_grad, direct_grad = torch.autograd.grad(state_only, (state, action), allow_unused=True)
        self.assertAlmostEqual(float(state_grad), 3.96, places=6)
        self.assertIsNone(direct_grad)


if __name__ == "__main__":
    unittest.main()
