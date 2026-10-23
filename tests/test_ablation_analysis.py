import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ablation.analyze import aggregate_curves, collect_records


def evaluation(seed=0, iteration=1):
    cfg = dict(subset="locomotion", horizon=32, hist=10, goal_frames=10, gamma=.99, lam=.95,
               amp="none", num_envs=32, pi_batch=32)
    return dict(method="ppo_sim", seed=seed, iteration=iteration, physical_return=.1 * iteration,
                tracking_failure_rate=1., protocol="fixed", evaluation_gpu="test-gpu", evaluation_seconds=10.,
                checkpoint_sha256=f"seed{seed}-it{iteration}", manifest=dict(config=cfg, gpu="test-gpu", torch="test",
                cuda="test", motions={"walk": "hash"}, assets={"robot": "hash"}, source_hashes={"supermimic/mdp.py": "same"}),
                train_ledger=dict(train_wall_seconds=100. * iteration, train_gpu_hours=100. * iteration / 3600,
                                  simulator_steps=320 * iteration, imagined_steps=0, wm_training_seconds=0.,
                                  policy_optimization_seconds=40. * iteration, collection_seconds=20. * iteration))


class AnalysisTests(unittest.TestCase):
    def collect(self, records):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for i, record in enumerate(records):
                path = Path(root) / f"{i}.json"
                path.write_text(json.dumps(record))
                paths.append(path)
            return collect_records(paths)

    def test_includes_all_prior_evaluations(self):
        rows = self.collect([evaluation(iteration=2), evaluation(iteration=1)])
        self.assertEqual([r["end_to_end_seconds"] for r in rows], [110., 220.])
        self.assertAlmostEqual(rows[1]["total_gpu_hours"], 220 / 3600)

    def test_different_gpu_or_motion_is_not_a_controlled_curve(self):
        first = evaluation()
        for field in ("gpu", "motions", "assets"):
            other = evaluation(seed=1)
            other["manifest"][field] = "different"
            with self.assertRaisesRegex(ValueError, "Mixed"):
                self.collect([first, other])

    def test_duplicate_evaluations_cannot_be_counted_as_seeds(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.collect([evaluation(), evaluation()])

    def test_single_seed_does_not_claim_zero_variance(self):
        rows = self.collect([evaluation()])
        curve = aggregate_curves(rows, "end_to_end_seconds")
        self.assertIsNone(curve[0]["std"])

    def test_mean_and_sample_standard_deviation(self):
        a, b, c = evaluation(0), evaluation(1), evaluation(2)
        a["physical_return"], b["physical_return"], c["physical_return"] = .2, .4, .6
        curve = aggregate_curves(self.collect([a, b, c]), "end_to_end_seconds")
        self.assertAlmostEqual(curve[0]["mean"], .4)
        self.assertAlmostEqual(curve[0]["std"], .2)


if __name__ == "__main__":
    unittest.main()
