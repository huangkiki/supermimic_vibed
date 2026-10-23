from pathlib import Path
import tempfile
import unittest

from experiments.ablation.run_suite import campaign_scope, check_forecast, source_snapshot
from experiments.ablation.validate_run import validate_run_layout


class CampaignIdentityTests(unittest.TestCase):
    def test_empty_or_partially_missing_campaign_cannot_pass_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = dict(args=dict(methods=["ppo_sim"], seeds=[0, 1]))
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_run_layout(root, campaign)
            for seed in (0, 1):
                p = root / f"ppo_sim-seed{seed}/manifest.json"
                p.parent.mkdir()
                p.write_text("{}")
                if seed == 0:
                    with self.assertRaisesRegex(ValueError, "coverage"):
                        validate_run_layout(root, campaign)
            self.assertEqual(len(validate_run_layout(root, campaign)), 2)

    def test_subset_tuning_keeps_pretraining_dependency_and_valid_bounds(self):
        scope = campaign_scope(["ppo_sim"], 1000, 500, 5e-6, 3)
        self.assertEqual(scope["methods"], ["ppo_sim"])
        self.assertEqual(scope["evaluation_repeats"], 3)
        self.assertEqual(campaign_scope(["ppo_wm_frozen", "pretrain"])["methods"], ["pretrain", "ppo_wm_frozen"])
        for args in [(["ppo_wm_frozen"],), (["ppo_sim", "ppo_sim"],),
                     (["ppo_sim"], 0), (["ppo_sim"], 1, 1, float("nan")),
                     (["ppo_sim"], 1, 1, 5e-6, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                campaign_scope(*args)

    def test_forecast_cannot_authorize_a_different_matrix_or_exhaust_resources(self):
        resources = dict(name="A100", uuid="gpu-1", memory_free_mib="40000")
        forecast = dict(gpu="A100", gpu_uuid="gpu-1", subset="locomotion", seeds=[0],
                        projected_campaign_gpu_hours=13.5, projected_peak_memory_mib=28000,
                        projected_disk_gib=12, minimum_disk_reserve_gib=12)
        check_forecast(forecast, resources, "locomotion", [0], 14, 50)
        scope = campaign_scope(["ppo_sim"], 1000, 500, 5e-6, 3)
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            check_forecast(forecast, resources, "locomotion", [0], 14, 50, scope)
        check_forecast(dict(forecast, scope=scope), resources, "locomotion", [0], 14, 50, scope)
        for changes in (dict(seeds=[0, 1, 2]), dict(gpu_uuid="gpu-2"), dict(subset="all40"),
                        dict(projected_campaign_gpu_hours=15), dict(projected_campaign_gpu_hours=float("nan")),
                        dict(projected_peak_memory_mib=39000), dict(projected_disk_gib=45)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_forecast(dict(forecast, **changes), resources, "locomotion", [0], 14, 50)

    def test_training_source_and_config_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "supermimic/train.py"
            config = root / "experiments/ablation/configs/test.json"
            code.parent.mkdir(parents=True)
            config.parent.mkdir(parents=True)
            code.write_text("version = 1\n")
            config.write_text('{"seed": 0}')
            initial = source_snapshot(root)
            (root / "log.txt").write_text("unrelated runtime output")
            self.assertEqual(source_snapshot(root), initial)
            code.write_text("version = 2\n")
            self.assertNotEqual(source_snapshot(root), initial)
            code.write_text("version = 1\n")
            config.write_text('{"seed": 1}')
            self.assertNotEqual(source_snapshot(root), initial)


if __name__ == "__main__":
    unittest.main()
