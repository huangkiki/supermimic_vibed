import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiments.ablation.analyze_policy_fidelity import compare


class NativeComponentPublicationTests(unittest.TestCase):
    def test_published_fp32_grid_and_incremental_costs_reconcile(self):
        root = Path(__file__).resolve().parents[1] / 'results/ablation/policy-gradient-components-v1'
        proof = json.loads((root / 'validation.json').read_text())
        archive = root / 'raw-data.tar.gz'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), proof['archive_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with tarfile.open(archive) as stream:
                names = set()
                for member in stream.getmembers():
                    name = Path(member.name)
                    self.assertTrue(member.isfile())
                    self.assertFalse(name.is_absolute() or '..' in name.parts)
                    data = stream.extractfile(member).read()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), proof['small_input_sha256'][member.name])
                    target = raw / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    names.add(member.name)
            self.assertEqual(names, set(proof['small_input_sha256']))
            probes = compare(sorted(p.parent for p in raw.glob('*/probe.json')), verify_tensors=False)
            self.assertEqual(len(probes), 4)
            self.assertEqual(sum(len(p['rows']) for p in probes), 120)
            self.assertTrue(all(not p['receipt']['precision']['tf32'] for p in probes))
            costs = json.loads((raw / 'h32-fp32-ablations-campaign.json').read_text())
            self.assertEqual(costs['status'], 'complete')
            self.assertEqual(len(costs['commands']), 3)  # Full-target probe reused, never paid twice.
            self.assertTrue(all(c['returncode'] == 0 for c in costs['commands']))
            seconds = sum(c['seconds'] for c in costs['commands'])
            self.assertEqual(seconds, proof['new_probe_seconds'])
            self.assertAlmostEqual((seconds + costs['prior_allocated_seconds']) / 3600,
                                   proof['cumulative_diagnostic_gpu_hours'])
            self.assertLess(proof['cumulative_diagnostic_gpu_hours'], .5)
            self.assertEqual(proof['all_prior_probes_strongly_revalidated'], 18)
            self.assertFalse(proof['cuda_initialized'])


if __name__ == '__main__':
    unittest.main()
