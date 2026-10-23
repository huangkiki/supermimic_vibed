import unittest

from experiments.ablation.common import Config


class ConfigTests(unittest.TestCase):
    def test_frozen_method_requires_an_identified_pretrain(self):
        with self.assertRaisesRegex(ValueError, "SAME pretrain"):
            Config(method="ppo_wm_frozen").validate()

    def test_sim_baseline_cannot_accidentally_inherit_model_cost(self):
        with self.assertRaisesRegex(ValueError, "PPO-Sim"):
            Config(method="ppo_sim", bootstrap="wrong.pt").validate()

    def test_nonfinite_budget_or_invalid_discount_rejected(self):
        for kwargs in ({"max_wall_seconds": float("inf")}, {"lr": float("nan")},
                       {"gamma": 1.}, {"lam": -0.1}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()

    def test_original_fractional_collection_schedule_is_supported(self):
        self.assertEqual(Config(collect_steps=2.5).validate().collect_steps, 2.5)


if __name__ == "__main__":
    unittest.main()
