"""Recompute the physical checkpoint study from published native evidence."""
import csv
import json
import math
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiments.ablation.analyze_checkpoint_audit import analyze, validate_native
from experiments.ablation.analyze_calibrated_updates import analyze as analyze_calibrated
from experiments.ablation.common import sha256

PUBLICATION = Path(__file__).resolve().parents[1] / 'results/ablation/checkpoint-audit-v1'


class NativeCheckpointPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.raw = Path(cls.temp.name)
        cls.manifest = json.loads((PUBLICATION / 'artifact-manifest.json').read_text())
        archive = PUBLICATION / 'raw-data.tar.gz'
        if sha256(archive) != cls.manifest['archive_sha256']:
            raise ValueError('Published archive identity changed')
        with tarfile.open(archive) as stream:
            stream.extractall(cls.raw, filter='data')

    def test_all_archived_files_and_retained_vectors_have_verified_identity(self):
        actual = {str(p.relative_to(self.raw)) for p in self.raw.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(self.manifest['published_files']))
        for name, expected in self.manifest['published_files'].items():
            path = self.raw / name
            self.assertEqual(sha256(path), expected['sha256'])
            self.assertEqual(path.stat().st_size, expected['bytes'])
        for phase in ('qualification', 'experiment'):
            saved = json.loads((self.raw / f'primary/cpu-validation/{phase}.json').read_text())
            self.assertTrue(saved['full_means_verified'])
            self.assertEqual(saved['full_mean_vectors'], 11)
            expected = self.manifest['retained_large_files'][f'primary/{phase}/gradient-means.pt']
            self.assertEqual(saved['gradient_means_sha256'], expected['sha256'])

    def test_reference_reuse_and_full_vector_validation_receipt(self):
        original = self.raw / 'reference-attempt/experiment'
        resumed = self.raw / 'calibrated/experiment'
        for name in ('reference-gradient-draws.json', 'reference-gradient-summary.json',
                     'reference-error-differences.json'):
            self.assertEqual((original / name).read_bytes(), (resumed / name).read_bytes())
        receipt = json.loads((self.raw / 'reference-attempt/cpu-validation.json').read_text())
        self.assertEqual(receipt['status'], 'passed')
        self.assertFalse(receipt['cuda_initialized'])
        self.assertEqual(receipt['full_parameter_count'], 9116477)
        self.assertEqual(len(receipt['error_difference_checks']), 14)
        retained = self.manifest['retained_large_files'][
            'reference-attempt/experiment/reference-gradient-means.pt'
        ]
        self.assertEqual(receipt['reference_means_sha256'], retained['sha256'])
        summary = json.loads((resumed / 'calibrated-summary.json').read_text())
        self.assertTrue(summary['reference_reused'])

    def test_primary_csv_and_native_step_evidence_recompute_without_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            analyze(self.raw / 'primary/experiment', output)
            validate_native(self.raw / 'primary/experiment')
            for path in (PUBLICATION / 'analysis').glob('*.csv'):
                self.assertEqual((output / path.name).read_bytes(), path.read_bytes())

    def test_calibrated_kl_and_raw_physical_improvements_recompute(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            analyze_calibrated(self.raw / 'calibrated/experiment', output,
                               parent=self.raw / 'primary/experiment')
            for path in (PUBLICATION / 'calibrated-analysis').glob('*.csv'):
                self.assert_derived_csv_equal(output / path.name, path)

    def assert_derived_csv_equal(self, actual, expected):
        """Allow FP64 reduction rounding across CPUs; identities stay exact."""
        with actual.open() as stream:
            actual_rows = list(csv.reader(stream))
        with expected.open() as stream:
            expected_rows = list(csv.reader(stream))
        self.assertEqual(actual_rows[0], expected_rows[0])
        self.assertEqual(len(actual_rows), len(expected_rows))
        for actual_row, expected_row in zip(actual_rows[1:], expected_rows[1:]):
            self.assertEqual(len(actual_row), len(expected_row))
            for value, reference in zip(actual_row, expected_row):
                try:
                    number, expected_number = float(value), float(reference)
                except ValueError:
                    self.assertEqual(value, reference)
                else:
                    self.assertTrue(
                        math.isclose(number, expected_number, rel_tol=1e-13, abs_tol=1e-15),
                        f'{actual.name}: {value} != {reference}',
                    )

    def test_derived_csv_allows_roundoff_but_rejects_changed_values_or_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            expected = Path(temp) / 'expected.csv'
            actual = Path(temp) / 'actual.csv'
            expected.write_text('method,value\nBPTT,1.0\n')
            actual.write_text('method,value\nBPTT,1.00000000000001\n')
            self.assert_derived_csv_equal(actual, expected)
            for changed in ('BPTT,1.00000001', 'PPO,1.0'):
                actual.write_text(f'method,value\n{changed}\n')
                with self.assertRaises(AssertionError):
                    self.assert_derived_csv_equal(actual, expected)

    def test_all_allocations_include_failure_and_share_one_cap(self):
        campaigns = {
            name: json.loads((self.raw / name / 'campaign.json').read_text())
            for name in ('failed-attempt', 'primary', 'reference-attempt', 'calibrated')
        }
        records = [row for campaign in campaigns.values() for row in campaign['commands']]
        calibrated = campaigns['calibrated']
        self.assertEqual([row['returncode'] for row in records], [1, 0, 0, 1, 0])
        total = sum(row['seconds'] for row in records) / 3600
        self.assertAlmostEqual(total, calibrated['combined_audit_gpu_hours'])
        self.assertLessEqual(total, 1)
        self.assertEqual(sha256(self.raw / 'primary/campaign.json'),
                         calibrated['parent_campaign_sha256'])
        self.assertEqual(sha256(self.raw / 'reference-attempt/campaign.json'),
                         calibrated['resumed_campaign_sha256'])
        self.assertAlmostEqual(calibrated['prior_audit_gpu_hours'],
                               sum(row['seconds'] for row in records[:-1]) / 3600)
        with (PUBLICATION / 'resource-accounting.csv').open() as stream:
            published = list(csv.DictReader(stream))
        self.assertEqual(len(published), 5)
        for row, record in zip(published, records):
            self.assertEqual(float(row['seconds']), record['seconds'])
            self.assertEqual(int(row['returncode']), record['returncode'])
            self.assertAlmostEqual(float(row['allocated_gpu_hours']), record['seconds'] / 3600)
        self.assertAlmostEqual(sum(float(row['allocated_gpu_hours']) for row in published), total)


if __name__ == '__main__':
    unittest.main()
