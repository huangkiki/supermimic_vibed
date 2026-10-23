import unittest
import json
from pathlib import Path

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

    def test_frozen_hard_comparator_cannot_silently_use_soft_objective(self):
        with self.assertRaisesRegex(ValueError, "hard termination"):
            Config(method="ppo_wm_frozen_hard", bootstrap="shared.pt").validate()
        Config(method="ppo_wm_frozen_hard", bootstrap="shared.pt", ppo_wm_survival="hard").validate()

    def test_campaign_configs_are_valid_and_share_frozen_pair_settings(self):
        root = Path(__file__).resolve().parents[1] / "experiments/ablation/configs"
        for phase in ("smoke", "pilot", "formal"):
            configs = {}
            for path in root.glob(f"{phase}-*.json"):
                raw = json.loads(path.read_text())
                if "_frozen" in raw["method"]:
                    raw["bootstrap"] = "shared.pt"
                configs[raw["method"]] = Config(**raw).validate()
            self.assertEqual(len(configs), 7)
            for key in ("horizon", "num_envs", "pi_batch", "hist", "goal_frames", "gamma", "lam", "ppo_wm_survival"):
                self.assertEqual(getattr(configs["ppo_wm_frozen"], key), getattr(configs["bptt_wm_frozen"], key))


if __name__ == "__main__":
    unittest.main()
