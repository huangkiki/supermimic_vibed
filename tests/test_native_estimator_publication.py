"""Reconcile published physical outcomes without loading multi-GB gradient vectors."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
import tempfile
import unittest

import torch

ROOT = Path(__file__).parents[1]/'results/ablation/matched-policy-v1'


class NativeEstimatorPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        with tarfile.open(ROOT/'raw-data.tar.gz') as archive:
            for item in archive.getmembers():
                path = Path(item.name)
                if not item.isfile() or path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe experiment archive')
            archive.extractall(cls.root,filter='data')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_corrected_cases_reconcile_physical_sampling_statistics_and_bank_identity(self):
        proof = json.loads((self.root/'all3-tensor-validation.json').read_text())
        self.assertEqual(proof['status'],'passed')
        self.assertFalse(proof['cuda_initialized'])
        self.assertEqual(len(proof['probes']),3)
        original = None
        row_count = 0
        for record in proof['probes']:
            path = self.root/record['probe']
            for name,identity in record['files'].items():
                candidate = path/name
                if candidate.exists():
                    self.assertEqual(candidate.stat().st_size,identity['bytes'])
                    self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(),identity['sha256'])
                else:
                    self.assertTrue(name.startswith('estimate-') or name=='mean-gradients.pt')
            bank = torch.load(path/'policy-bank.pt',map_location='cpu',weights_only=False)
            if original is None:
                original = bank
            for key in ('state','frames','xi','previous_action'):
                torch.testing.assert_close(original[key],bank[key],rtol=0,atol=0)
            for key in ('s','f','xi','a_prev'):
                torch.testing.assert_close(original['replay_history'][key],bank['replay_history'][key],rtol=0,atol=0)
            summary = json.loads((path/'comparison-summary.json').read_text())
            self.assertEqual(summary['action_std'],.1)
            self.assertEqual(len(summary['gradient_seeds']),16)
            self.assertEqual(len(summary['reference_seeds']),16)
            self.assertFalse(set(summary['gradient_seeds'])&set(summary['reference_seeds']))
            raw = torch.load(path/'physical-reference.pt',map_location='cpu',weights_only=False)
            with (path/'comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows,[{key:str(value) for key,value in r.items()} for r in raw['rows']])
            self.assertEqual(len(rows),12)
            row_count += len(rows)
            for row in rows:
                epsilon = float(row['epsilon'])
                scan = next(s for s in raw['perturbations'][row['direction']] if s['epsilon']==epsilon)
                objective = row['objective']
                derivatives = [float(p[objective].mean()-m[objective].mean())/(2*epsilon)
                               for p,m in zip(scan['plus'],scan['minus'])]
                improvements = [float(p[objective].mean()-b[objective].mean())
                                for p,b in zip(scan['plus'],raw['baseline'])]
                for label,values in (('derivative',derivatives),('improvement',improvements)):
                    self.assertEqual(len(values),16)
                    self.assertAlmostEqual(float(row[f'{label}_mean']),statistics.mean(values),places=10)
                    self.assertAlmostEqual(float(row[f'{label}_standard_error']),statistics.stdev(values)/4,places=10)
        self.assertEqual(row_count,36)

    def test_failures_and_numerical_audit_remain_in_the_cumulative_cost(self):
        first = json.loads((self.root/'h1-allocation.json').read_text())
        initial = json.loads((self.root/'horizon-campaign.json').read_text())
        audit = json.loads((self.root/'score-audit-allocation.json').read_text())
        corrected = json.loads((self.root/'corrected-campaign.json').read_text())
        ledger = json.loads((self.root/'cost-reconciliation.json').read_text())
        commands = [first,*initial['commands'],audit,*corrected['commands']]
        self.assertEqual(len(commands),7)
        self.assertEqual(sum(c['returncode']!=0 for c in commands),1)
        self.assertEqual(initial['status'],'failed')
        self.assertEqual(corrected['status'],'complete')
        hours = sum(c['seconds'] for c in commands)/3600
        self.assertAlmostEqual(hours,ledger['matched_comparison_gpu_hours'])
        self.assertAlmostEqual(hours+first['prior_diagnostic_gpu_hours'],ledger['cumulative_diagnostic_gpu_hours'])
        self.assertLess(ledger['cumulative_diagnostic_gpu_hours'],ledger['cumulative_cap_gpu_hours'])
        self.assertEqual(len({c['resources']['uuid'] for c in commands}),1)
        numerical = json.loads((self.root/'score-batch-audit.json').read_text())
        self.assertEqual(numerical['parameter_sha256_before'],numerical['parameter_sha256_after'])
        self.assertGreater(numerical['flat_max_logprob_error'],1e-4)
        self.assertEqual(numerical['sequential_max_logprob_error'],0)
        self.assertTrue(numerical['sequential_bitwise_equal'])


if __name__=='__main__':
    unittest.main()
