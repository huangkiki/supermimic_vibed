from pathlib import Path
import tempfile
import unittest

from experiments.ablation.run_suite import source_snapshot


class CampaignIdentityTests(unittest.TestCase):
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
