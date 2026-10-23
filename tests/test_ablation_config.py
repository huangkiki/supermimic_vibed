import unittest
import json
from pathlib import Path

import torch

from experiments.ablation.common import Config, set_learning_rates


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

    def test_actor_rate_survives_schedule_without_changing_model_or_critic_rate(self):
        cfg = Config(actor_lr=5e-6, lr=2e-4, lr_decay_iters=100).validate()
        parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
        optimizers = [torch.optim.SGD([p], lr=1.) for p in parameters]
        rates = set_learning_rates(cfg, 51, *optimizers)
        for parameter, optimizer in zip(parameters, optimizers):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        self.assertAlmostEqual(rates["actor_lr"], 5e-6 * .515)
        self.assertAlmostEqual(rates["lr"], 2e-4 * .515)
        self.assertAlmostEqual(float(parameters[0]), 1 - rates["actor_lr"], places=7)
        for parameter in parameters[1:]:
            self.assertAlmostEqual(float(parameter), 1 - rates["lr"], places=7)
        rates = set_learning_rates(cfg, 1001, *optimizers)
        self.assertAlmostEqual(rates["actor_lr"] / rates["lr"], .025)
        self.assertAlmostEqual(rates["lr"], cfg.lr * .03)

    def test_legacy_rate_and_absent_world_model_are_supported(self):
        cfg = Config().validate()
        optimizers = [torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.) for _ in range(2)]
        rates = set_learning_rates(cfg, 1, *optimizers)
        self.assertEqual(rates, {"lr": cfg.lr, "actor_lr": cfg.lr})
        self.assertEqual([o.param_groups[0]["lr"] for o in optimizers], [cfg.lr, cfg.lr])

    def test_invalid_actor_rate_is_rejected(self):
        for rate in (0., -1., float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                Config(actor_lr=rate).validate()

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
