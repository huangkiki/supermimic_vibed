"""Recompute the published native metadata; no native GPU work runs in CI."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import torch

from experiments.ablation.analyze_policy_fidelity import compare, load_probe

ROOT=Path(__file__).parents[1]/'results/ablation/policy-gradient-native-v1'


class NativePolicyPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        with tarfile.open(ROOT/'raw-data.tar.gz') as archive:
            for member in archive.getmembers():
                path=Path(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe publication archive')
            archive.extractall(cls.root,filter='data')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_native_measurements_recompute_and_fixed_histories_match(self):
        proof=json.loads((self.root/'all15-tensor-validation.json').read_text())
        self.assertEqual(len(proof['probes']),15)
        count=0
        original=None
        for record in proof['probes']:
            path=self.root/record['probe']
            for name in ('policy-bank.pt','directions.csv','summary.json','probe.json'):
                self.assertEqual(hashlib.sha256((path/name).read_bytes()).hexdigest(),record['files'][name]['sha256'])
            probe=load_probe(path,verify_tensors=False)
            count+=len(probe['rows'])
            bank=torch.load(path/'policy-bank.pt',map_location='cpu',weights_only=False)
            if original is None:original=bank
            for key in ('state','frames','xi','previous_action'):
                torch.testing.assert_close(original[key],bank[key],rtol=0,atol=0)
            for key in ('s','a_prev','f','xi'):
                torch.testing.assert_close(original['replay_history'][key],bank['replay_history'][key],rtol=0,atol=0)
        self.assertEqual(count,450)
        self.assertEqual(original['motion_names'],['run1_subject2','run1_subject5'])

    def test_native_precision_groups_stay_separate_and_full_costs_reconcile(self):
        with self.assertRaisesRegex(ValueError,'precision'):
            compare([self.root/'h1-full-01',self.root/'h8-full-fp32-01'],verify_tensors=False)
        first=json.loads((self.root/'h1-full-01-allocation.json').read_text())
        horizons=json.loads((self.root/'horizon-campaign.json').read_text())
        ablation=json.loads((self.root/'ablation-campaign.json').read_text())
        precision=json.loads((self.root/'precision-campaign.json').read_text())
        commands=[first,*horizons['commands'],*ablation['commands'],*precision['commands']]
        self.assertEqual(len(commands),15)
        self.assertTrue(all(c['returncode']==0 for c in commands))
        self.assertAlmostEqual(sum(c['seconds'] for c in commands)/3600,precision['allocated_gpu_hours'])
        self.assertLess(precision['allocated_gpu_hours'],precision['max_total_gpu_hours'])
        self.assertEqual({c['resources']['uuid'] for c in commands},{precision['gpu_uuid']})


if __name__=='__main__':
    unittest.main()
