"""Recompute the published native experiment without CUDA or large gradient files."""
import json
import csv
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiments.ablation.analyze_estimator_convergence import reference_audit, validate
from experiments.ablation.common import sha256
from experiments.ablation.estimator_convergence import analyze


PUBLICATION = Path(__file__).resolve().parents[1] / 'results/ablation/estimator-convergence-v1'


class NativeConvergencePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.temporary.name)
        cls.manifest = json.loads((PUBLICATION / 'artifact-manifest.json').read_text())
        archive = PUBLICATION / 'raw-data.tar.gz'
        if sha256(archive) != cls.manifest['archive_sha256']:
            raise ValueError('Publication archive hash differs')
        with tarfile.open(archive) as stream:
            stream.extractall(cls.raw, filter='data')
        cls.campaign = json.loads((cls.raw / 'campaign.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_published_bytes_match_manifest_and_frozen_inputs(self):
        actual = {str(p.relative_to(self.raw)) for p in self.raw.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(self.manifest['published_files']))
        for name, expected in self.manifest['published_files'].items():
            path = self.raw / name
            self.assertEqual(path.stat().st_size, expected['bytes'])
            self.assertEqual(sha256(path), expected['sha256'])
        self.assertEqual(self.campaign['source'], self.manifest['native_source'])
        for phase in ('qualification', 'experiment'):
            receipt = json.loads((self.raw / phase / 'probe.json').read_text())
            for name in ('checkpoint', 'bank'):
                self.assertEqual(receipt[name + '_sha256'], self.campaign['plan'][name + '_sha256'])
            self.assertEqual(receipt['resources']['uuid'], self.campaign['plan']['gpu_uuid'])

    def test_every_published_reference_and_projection_recomputes_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            for phase in ('qualification', 'experiment'):
                raw, summary, costs, variance = validate(self.raw / phase)
                dest = Path(directory) / phase
                analyze(raw, dest)
                for name in ('convergence.csv', 'directional-comparison.csv'):
                    self.assertEqual((dest / name).read_bytes(), (self.raw / phase / name).read_bytes())
                design = self.campaign['plan'][phase]
                self.assertEqual(summary['counts'], design['counts'])
                self.assertEqual(summary['directions'], design['directions'])
                self.assertEqual(summary['gradient_seeds'], list(range(design['gradient_start'], design['gradient_start'] + design['counts'][-1])))
                self.assertEqual(summary['reference_seeds'], list(range(design['reference_start'], design['reference_start'] + design['reference_count'])))

    def test_both_native_allocations_including_qualification_fit_original_cap(self):
        self.assertEqual(self.campaign['status'], 'complete')
        records = self.campaign['commands']
        self.assertEqual([r['phase'] for r in records], ['qualification', 'experiment'])
        for record in records:
            self.assertEqual(record['returncode'], 0)
            native = json.loads((self.raw / record['phase'] / 'probe.json').read_text())
            self.assertGreaterEqual(record['seconds'], native['seconds'])
            self.assertGreater(record['peak_device_memory_used_mib'], 0)
        allocation = sum(r['seconds'] for r in records)/3600
        self.assertAlmostEqual(allocation, self.campaign['allocated_gpu_hours'])
        self.assertLessEqual(allocation, self.campaign['plan']['hard_gpu_hours'])
        self.assertLessEqual(self.campaign['projected_experiment_seconds'], records[1]['maximum_seconds']-20)

    def test_posthoc_repeats_preserve_inputs_and_share_the_original_budget(self):
        archive=PUBLICATION/'reference-audit.tar.gz'
        self.assertEqual(sha256(archive),self.manifest['reference_audit_archive_sha256'])
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as analysis:
            audit=Path(directory)
            with tarfile.open(archive) as stream:
                stream.extractall(audit,filter='data')
            actual={str(p.relative_to(audit)) for p in audit.rglob('*') if p.is_file()}
            self.assertEqual(actual,set(self.manifest['reference_audit_published_files']))
            for name,expected in self.manifest['reference_audit_published_files'].items():
                self.assertEqual(sha256(audit/name),expected['sha256'])
                self.assertEqual((audit/name).stat().st_size,expected['bytes'])
            reference_audit(self.raw/'experiment',audit,analysis)
            self.assertEqual((Path(analysis)/'reference-audit.csv').read_bytes(),
                             (PUBLICATION/'analysis/reference-audit.csv').read_bytes())
            record=json.loads((audit/'campaign.json').read_text())
            self.assertEqual(len(record['commands']),3)
            self.assertTrue(all(c['returncode']==0 for c in record['commands']))
            total=self.campaign['allocated_gpu_hours']+sum(c['seconds'] for c in record['commands'])/3600
            self.assertAlmostEqual(record['combined_allocated_gpu_hours'],total)
            self.assertLessEqual(total,record['combined_cap_gpu_hours'])
            with (PUBLICATION/'resource-accounting.csv').open() as stream:
                costs=list(csv.DictReader(stream))
            self.assertEqual(len(costs),5)
            self.assertAlmostEqual(sum(float(r['allocated_gpu_hours']) for r in costs),total)


if __name__ == '__main__':
    unittest.main()
