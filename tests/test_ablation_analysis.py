import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ablation.analyze import aggregate_curves, collect_records, main
from experiments.ablation.plot_fidelity import main as fidelity_main


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
    def plot_prediction_performance(self, root, repeat_count):
        probe = root / 'probe'
        probe.mkdir()
        record = evaluation()
        sha = record['checkpoint_sha256']
        (probe / 'probe.json').write_text(json.dumps(dict(
            checkpoint_sha256={'ckpt_000001.pt': sha}, args=dict(samples=1, directions=1))))
        tables = {
            'prediction.csv': [dict(checkpoint_sha256=sha, sample=0, horizon=h, scaled_mse=.01 * h)
                               for h in (1, 8, 16, 32)],
            'gradient.csv': [dict(checkpoint_sha256=sha, gradient_horizon=1, reward_objective='full',
                                  contact_group_level='body', contact_group='stable_contact',
                                  objective=objective, direction_type=direction, resolved=True,
                                  fd_derivative=1., predicted_derivative=1., epsilon=.01,
                                  initial_state_max_drift=0., improvement=.01)
                             for objective in ('hard', 'soft')
                             for direction in ('wm_gradient', 'random_unit_l2')],
        }
        for name, rows in tables.items():
            with (probe / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        paths = []
        for repeat, score in enumerate((.2, .4, .9)[:repeat_count]):
            record = evaluation()
            record.update(evaluation_repeat_index=repeat, evaluation_repeat_count=3, physical_return=score)
            path = root / f'eval{repeat}.json'
            path.write_text(json.dumps(record))
            paths.append(str(path))
        output = root / 'plots'
        with patch('sys.argv', ['plot_fidelity', str(probe), '--output', str(output), '--evaluations', *paths]):
            fidelity_main()
        return output

    def test_prediction_performance_plot_uses_all_repeats_not_the_last_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.plot_prediction_performance(Path(directory), 3)
            row, = json.loads((output / 'prediction-performance.json').read_text())
            self.assertAlmostEqual(row['physical_return'], .5)
            self.assertAlmostEqual(row['evaluation_return_std'], .36055512754639896)
            self.assertEqual(row['evaluation_repeats'], 3)
            self.assertEqual(row['seed'], 0)
            for horizon in (1, 8, 16, 32):
                self.assertAlmostEqual(row[f'mse{horizon}'], .01 * horizon)
            self.assertTrue((output / 'prediction-performance.png').is_file())

    def test_prediction_performance_plot_rejects_missing_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing'):
                self.plot_prediction_performance(Path(directory), 2)

    def test_plot_entrypoint_reports_raw_repeat_count_after_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for iteration in (1, 2):
                for repeat in range(3):
                    record = evaluation(iteration=iteration)
                    record.update(evaluation_repeat_index=repeat, evaluation_repeat_count=3)
                    path = root / f"{iteration}-{repeat}.json"
                    path.write_text(json.dumps(record))
                    paths.append(str(path))
            output = root / "plots"
            with patch("sys.argv", ["analyze", "--evaluations", *paths, "--output", str(output)]):
                main()
            summary = json.loads((output / "analysis.json").read_text())
            self.assertEqual(summary["evaluation_count"], 6)
            self.assertEqual(summary["checkpoint_count"], 2)
            self.assertEqual(summary["seeds"], [0])
            self.assertTrue((output / "return-compute.png").exists())

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
        for field in ("gpu", "motions", "assets", "source_hashes"):
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

    def test_repeats_average_within_checkpoint_charge_every_evaluation_and_remain_one_seed(self):
        records = []
        for iteration in (1, 2):
            for index, score in enumerate((.2, .4, .6)):
                record = evaluation(iteration=iteration)
                record.update(evaluation_repeat_index=index, evaluation_repeat_count=3,
                              physical_return=score, evaluation_simulator_steps=7)
                records.append(record)
        rows = self.collect(records)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["end_to_end_seconds"] for r in rows], [130., 260.])
        self.assertEqual([r["evaluation_simulator_steps"] for r in rows], [21, 42])
        self.assertAlmostEqual(rows[0]["physical_return"], .4)
        self.assertAlmostEqual(rows[0]["evaluation_return_std"], .2)
        self.assertEqual(rows[0]["evaluation_repeats"], 3)
        curve = aggregate_curves(rows, "end_to_end_seconds")
        self.assertEqual(curve[0]["seeds"], 1)
        self.assertIsNone(curve[0]["std"])

    def test_missing_duplicate_and_wrong_checkpoint_repeats_are_rejected(self):
        a, b = evaluation(), evaluation()
        a.update(evaluation_repeat_index=0, evaluation_repeat_count=2)
        b.update(evaluation_repeat_index=1, evaluation_repeat_count=2)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.collect([a])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.collect([a, a])
        b["checkpoint_sha256"] = "not-the-same-policy"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.collect([a, b])

    def test_same_gpu_model_but_different_physical_device_is_not_silently_mixed(self):
        a, b = evaluation(), evaluation(seed=1)
        for record, uuid in ((a, "gpu-a"), (b, "gpu-b")):
            record["manifest"]["gpu_uuid"] = uuid
            record["evaluation_gpu_uuid"] = uuid
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.collect([a, b])
        a["evaluation_gpu_uuid"] = "gpu-b"
        with self.assertRaisesRegex(ValueError, "GPUs differ"):
            self.collect([a])

    def test_mean_and_sample_standard_deviation(self):
        a, b, c = evaluation(0), evaluation(1), evaluation(2)
        a["physical_return"], b["physical_return"], c["physical_return"] = .2, .4, .6
        curve = aggregate_curves(self.collect([a, b, c]), "end_to_end_seconds")
        self.assertAlmostEqual(curve[0]["mean"], .4)
        self.assertAlmostEqual(curve[0]["std"], .2)

    def test_final_endpoint_includes_process_and_shared_pretrain_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "ppo_wm_frozen-seed0"
            pretrain = root / "pretrain-seed0"
            run.mkdir()
            pretrain.mkdir()
            (run / "completion.json").write_text(json.dumps(dict(iterations=1)))
            (pretrain / "completion.json").write_text(json.dumps(dict(iterations=1, ledger=dict(train_wall_seconds=30.))))
            record = evaluation()
            record["method"] = "ppo_wm_frozen"
            path = run / "evaluation.json"
            path.write_text(json.dumps(record))
            commands = [dict(command=["python", "-m", "experiments.ablation.train", "--output", str(p)],
                             seconds=seconds, returncode=0) for p, seconds in ((run, 105.), (pretrain, 35.))]
            commands.append(dict(command=["python", "-m", "experiments.ablation.evaluate", "--checkpoint", str(run / "ckpt_000001.pt")],
                                 seconds=12., returncode=0))
            campaign = root / "campaign.json"
            campaign.write_text(json.dumps(dict(commands=commands)))
            row, = collect_records([path], campaign)
            self.assertEqual(row["end_to_end_seconds"], 152.)
            self.assertEqual(row["cost_basis"], "complete_process_allocation")

    def test_repeated_evaluations_each_charge_their_own_process_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "ppo_sim-seed0"
            run.mkdir()
            (run / "completion.json").write_text(json.dumps(dict(iterations=1)))
            commands = [dict(command=["python", "-m", "experiments.ablation.train", "--output", str(run)],
                             seconds=120., returncode=0)]
            paths = []
            for index in range(3):
                record = evaluation()
                record.update(evaluation_repeat_index=index, evaluation_repeat_count=3)
                path = run / f"repeat{index}.json"
                path.write_text(json.dumps(record))
                paths.append(path)
                commands.append(dict(command=["python", "-m", "experiments.ablation.evaluate", "--checkpoint",
                    str(run / "ckpt_000001.pt"), "--repeat-index", str(index)], seconds=12.+index, returncode=0))
            campaign = root / "campaign.json"
            campaign.write_text(json.dumps(dict(commands=commands)))
            row, = collect_records(paths, campaign)
            self.assertEqual(row["end_to_end_seconds"], 159.)
            self.assertEqual(row["cumulative_evaluation_seconds"], 39.)


if __name__ == "__main__":
    unittest.main()
