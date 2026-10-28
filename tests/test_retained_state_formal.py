import tempfile
import unittest
from pathlib import Path

from experiments.retained_state.run_formal import (
    CONDITIONS,
    evaluate_gate,
    evaluation_command,
    load_checkpoint_metrics,
    parse_evaluation_log,
    training_command,
)


class RetainedStateFormalTests(unittest.TestCase):
    def test_commands_pin_training_budget_and_candidate_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = training_command(
                "python", run_dir=root / "original", motion_dir=root / "data",
                seed=2, iterations=4000, condition="original",
            )
            retained = training_command(
                "python", run_dir=root / "retained", motion_dir=root / "data",
                seed=2, iterations=4000, condition="retained",
            )
            self.assertEqual(CONDITIONS["original"], ())
            self.assertIn("--wm_state_skip", retained)
            self.assertEqual(retained[retained.index("--wm_latent_dim") + 1], "128")
            self.assertNotIn("--wm_state_skip", original)
            for flag, value in (
                ("--iterations", "4000"), ("--num_envs", "4096"),
                ("--wm_batch", "2048"), ("--pi_batch", "4096"),
                ("--wm_window", "8"), ("--horizon", "32"), ("--seed", "2"),
            ):
                self.assertEqual(original[original.index(flag) + 1], value)
            evaluation = evaluation_command(
                "python", run_dir=root / "retained",
                checkpoint=root / "retained/ckpt_004000.pt",
                metrics_path=root / "retained/ckpt_004000_eval.json",
                motion_dir=root / "data",
            )
            self.assertIn("--ckpt", evaluation)
            self.assertIn("--metrics-json", evaluation)

    def test_gate_requires_gain_failures_and_two_positive_seeds(self):
        results = {}
        for seed, baseline, candidate, base_fail, candidate_fail in (
            (2, 0.70, 0.75, 2, 1),
            (3, 0.71, 0.74, 3, 2),
            (4, 0.69, 0.68, 1, 2),
        ):
            results[f"original-seed{seed}"] = {
                "norm_return": baseline, "n_failed": base_fail
            }
            results[f"retained-seed{seed}"] = {
                "norm_return": candidate, "n_failed": candidate_fail
            }
        gate = evaluate_gate(results)
        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(gate["relative_return_gain"], 0.03)
        self.assertEqual(gate["positive_seeds"], 2)
        results["retained-seed4"]["norm_return"] = 0.60
        gate = evaluate_gate(results)
        self.assertFalse(gate["passed"])

    def test_gate_rejects_incomplete_matrix(self):
        gate = evaluate_gate({})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["reason"], "incomplete")
        self.assertEqual(len(gate["missing"]), 6)

    def test_evaluation_metrics_are_parsed_from_mixed_simulator_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.log"
            path.write_text(
                "simulator startup {not json}\n"
                '{"norm_return": 0.7, "norm_return_framew": 0.71, '
                '"n_failed": 2, "failed": ["a", "b"]}\n'
                "[INFO] context cleared\n"
            )
            metrics = parse_evaluation_log(path)
            self.assertEqual(metrics["norm_return"], 0.7)
            self.assertEqual(metrics["failed"], ["a", "b"])

    def test_checkpoint_metrics_require_per_clip_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                '{"norm_return": 0.7, "norm_return_framew": 0.71, '
                '"n_failed": 1, "failed": ["a"]}'
            )
            with self.assertRaisesRegex(ValueError, "per_clip"):
                load_checkpoint_metrics(path)


if __name__ == "__main__":
    unittest.main()
