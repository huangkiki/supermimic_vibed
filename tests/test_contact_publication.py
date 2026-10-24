"""Recheck published small physical evidence; full gradient vectors stay hash-addressed."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import torch

from experiments.ablation.analyze_policy_fidelity import load_probe, validate_mean_difference
from experiments.ablation.analyze_policy_contacts import contact_evidence

ROOT = Path(__file__).parents[1] / 'results/ablation/policy-gradient-contacts-v1'


class ContactPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.proof = json.loads((ROOT / 'validation.json').read_text())
        archive = ROOT / 'raw-data.tar.gz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == cls.proof['archive_sha256']
        with tarfile.open(archive) as stream:
            if any(not item.isfile() or Path(item.name).is_absolute() or '..' in Path(item.name).parts for item in stream.getmembers()):
                raise ValueError('Unsafe archive member')
            stream.extractall(cls.root, filter='data')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_small_evidence_hashes_and_failed_cost_are_retained(self):
        for name, digest in self.proof['small_input_sha256'].items():
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest)
        failed = json.loads((self.root / 'failed/campaign.json').read_text())
        current = self.proof['native_campaign']
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(len(failed['commands']), 1)
        self.assertNotEqual(failed['commands'][0]['returncode'], 0)
        self.assertAlmostEqual(failed['cumulative_diagnostic_gpu_hours'] * 3600, current['prior_allocated_seconds'])
        self.assertAlmostEqual(current['cumulative_diagnostic_gpu_hours'],
            (current['prior_allocated_seconds'] + sum(c['seconds'] for c in current['commands'])) / 3600)
        self.assertLess(current['cumulative_diagnostic_gpu_hours'], current['cumulative_cap_gpu_hours'])
        self.assertFalse(self.proof['cuda_initialized'])
        self.assertTrue(self.proof['all_raw_vectors_and_physical_rollouts_verified'])

    def test_physical_return_measurements_and_contact_groups_recompute(self):
        root = self.root / 'successful'
        plan = json.loads((root / 'plan.json').read_text())
        mapping = json.loads((root / 'history64/contact-map.json').read_text())['geom_body_ids']
        total = 0
        observed = []
        for group, indices in plan['cases'].items():
            probe = load_probe(root / group, verify_tensors=False)
            bank = torch.load(root / group / 'policy-bank.pt', map_location='cpu', weights_only=False)
            raw = torch.load(self.root / 'physical' / f'{group}.pt', map_location='cpu', weights_only=False)
            self.assertNotIn('unit_directions', raw)
            self.assertEqual(raw['rows'], probe['rows'])
            observed.extend(dict(cohort=group, **row) for row in contact_evidence(bank, raw, mapping,
                group=group, indices=indices, motions=plan['motion_names'], horizon=32))
            for row in probe['rows']:
                plus, minus, *_ = raw['perturbations'][row['direction']][plan['epsilons'].index(row['epsilon'])]
                for name, value in [('positive_return', plus[row['objective']]), ('negative_return', minus[row['objective']])]:
                    validate_mean_difference(row[name], value, torch.zeros_like(value))
            for target in ('proxy', 'hard'):
                validate_mean_difference(probe['summary']['actual_adam_improvement'][target],
                                         raw['actual_adam'][target], raw['baselines'][0][target])
            total += len(probe['rows'])
        self.assertEqual(total, 132)
        with (ROOT / 'contacts.csv').open() as stream:
            published = list(csv.DictReader(stream))
        self.assertEqual(published, [{k: str(v) for k, v in row.items()} for row in observed])


if __name__ == '__main__':
    unittest.main()
