"""Regressions against the frozen physical experiment, requiring no simulator."""
import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest


class RecordedPPOPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / 'raw'
        self.output = Path(self.temp.name) / 'analysis'
        self.published = Path(__file__).resolve().parents[1] / 'results/ablation/confirmatory-seed1-ppo-pair-v1'
        spec = importlib.util.spec_from_file_location('recorded_ppo_pair', self.published / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reproduce = module.reproduce
        with tarfile.open(self.published / 'raw-data.tar.gz') as archive:
            for member in archive.getmembers():
                relative = Path(member.name).relative_to('confirmatory-seed1-ppo-pair-v1')
                self.assertTrue(member.isfile())
                self.assertFalse(relative.is_absolute())
                self.assertNotIn('..', relative.parts)
                target = self.raw / relative
                self.assertFalse(target.exists())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())

    def test_real_archive_reproduces_all_published_numeric_tables(self):
        summary = self.reproduce(self.raw, self.output)
        self.assertEqual(summary['physical_evaluations'], 30)
        self.assertEqual(summary['per_clip_rows'], 540)
        self.assertFalse(summary['cohort_complete'])
        tables = [p for p in self.output.iterdir() if p.suffix in ('.csv', '.json')]
        self.assertEqual(len(tables), 9)
        for table in tables:
            with self.subTest(table=table.name):
                self.assertEqual(table.read_bytes(), (self.published / table.name).read_bytes())

    def test_missing_final_physical_repeat_is_rejected_before_output(self):
        (self.raw / 'ppo_wm_online-seed1/ckpt_004000/repeat_002/evaluation.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.reproduce(self.raw, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
