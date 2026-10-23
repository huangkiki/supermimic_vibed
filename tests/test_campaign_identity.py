from pathlib import Path
import tempfile
import unittest

from experiments.ablation.run_suite import check_forecast, source_snapshot


class CampaignIdentityTests(unittest.TestCase):
    def test_forecast_cannot_authorize_a_different_matrix_or_exhaust_resources(self):
        resources = dict(name="A100", uuid="gpu-1", memory_free_mib="40000")
        forecast = dict(gpu="A100", gpu_uuid="gpu-1", subset="locomotion", seeds=[0],
                        projected_campaign_gpu_hours=13.5, projected_peak_memory_mib=28000,
                        projected_disk_gib=12, minimum_disk_reserve_gib=12)
        check_forecast(forecast, resources, "locomotion", [0], 14, 50)
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
