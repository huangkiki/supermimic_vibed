import unittest
import numpy as np
from experiments.ablation.fidelity_math import agreement, contact_groups, reference_quality


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


if __name__ == "__main__":
    unittest.main()
