import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_policy_fidelity import fixture
from experiments.ablation.common import sha256, write_json
from experiments.ablation.analyze import save_csv
from experiments.ablation.policy_fidelity import scan_parameter_directions, summarize_direction_scan
from experiments.ablation.analyze_policy_fidelity import compare, load_probe


def write_probe(path, horizon=1, detach=False):
    path.mkdir()
    probe, optimizer = fixture(horizon, detach_action_reward=detach)
    capture = probe.capture(optimizer)
    def reference():
        value = probe.objective()['proxy']
        return dict(proxy=value, hard=-value, initial_state_drift=torch.zeros(probe.batch))
    result = scan_parameter_directions(probe, capture, reference, epsilons=[.001,.003,.01], random_directions=2)
    bank = dict(state=probe.starts['s'],frames=probe.starts['f'],xi=probe.starts['xi'],
                previous_action=probe.starts['a_prev'],replay_history=probe.history)
    torch.save(bank,path/'policy-bank.pt')
    torch.save(capture,path/'gradients.pt')
    torch.save(result,path/'direction-rollouts.pt')
    save_csv(path/'directions.csv',result['rows'])
    write_json(path/'summary.json',summarize_direction_scan(result))
    write_json(path/'probe.json',dict(status='complete',checkpoint_sha256='fixed',
        args=dict(gradient_horizon=horizon,detach_action_reward=detach,no_critic=False,
                  sample_indices=[0,1],probe_seed=9001,directions=2,epsilons=[.001,.003,.01]),
        precision=dict(tf32=False,amp='None',compiled=False), reset_protocol='fixed',
        manifest=dict(gpu_uuid='GPU-one'),prepared_bank_sha256=sha256(path/'policy-bank.pt'), seconds=1.))
    return path


class PolicyAnalysisTests(unittest.TestCase):
    def test_complete_paired_probes_recompute_from_raw_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            results=compare([write_probe(root/'h1',1),write_probe(root/'h8',8)])
            self.assertEqual(len(results),2)
            self.assertTrue(all(r['tensors_verified'] for r in results))
            self.assertTrue(all(r['norms']['adam_displacement_l2']>0 for r in results))
            self.assertEqual(len(results[0]['per_start']),4)

    def test_missing_scale_duplicate_or_corrupted_measurement_is_rejected(self):
        for kind in ('missing','missing_direction','duplicate','derivative','sign','mislabel'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=write_probe(Path(directory)/'run')
                with (path/'directions.csv').open() as stream:
                    rows=list(csv.DictReader(stream))
                if kind=='missing':rows.pop()
                if kind=='missing_direction':rows=[r for r in rows if r['direction']!='random_1']
                if kind=='duplicate':rows.append(rows[-1].copy())
                if kind=='derivative':rows[0]['numerical_derivative']='12345'
                if kind=='sign':rows[0]['sign_agrees']='False' if rows[0]['sign_agrees']=='True' else 'True'
                if kind=='mislabel':rows[0]['interpretation']='update_direction_effectiveness_only'
                save_csv(path/'directions.csv',rows)
                with self.assertRaises(ValueError):load_probe(path)

    def test_different_gpu_precision_or_repeat_case_cannot_be_pooled(self):
        for kind in ('gpu','precision','repeat'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);a=write_probe(root/'a',1);b=write_probe(root/'b',1 if kind=='repeat' else 8)
                record=json.loads((b/'probe.json').read_text())
                if kind=='gpu':record['manifest']['gpu_uuid']='GPU-two'
                if kind=='precision':record['precision']['tf32']=True
                write_json(b/'probe.json',record)
                with self.assertRaises(ValueError):compare([a,b])

    def test_changed_raw_history_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=write_probe(root/'a',1);b=write_probe(root/'b',8)
            bank=torch.load(b/'policy-bank.pt',weights_only=False)
            bank['replay_history']['s'][0,0,0]+=1
            torch.save(bank,b/'policy-bank.pt')
            record=json.loads((b/'probe.json').read_text());record['prepared_bank_sha256']=sha256(b/'policy-bank.pt')
            write_json(b/'probe.json',record)
            with self.assertRaises(AssertionError):compare([a,b])

    def test_corrupted_raw_physical_return_or_restore_drift_is_rejected(self):
        for kind in ('return','drift'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=write_probe(Path(directory)/'run')
                raw=torch.load(path/'direction-rollouts.pt',weights_only=False)
                if kind=='return':raw['perturbations']['raw_gradient'][0][0]['proxy']+=1
                else:raw['baselines'][0]['initial_state_drift']+=.001
                torch.save(raw,path/'direction-rollouts.pt')
                with self.assertRaises(ValueError):load_probe(path)

    def test_direct_action_ablation_retains_effectiveness_label(self):
        with tempfile.TemporaryDirectory() as directory:
            result=load_probe(write_probe(Path(directory)/'run',detach=True))
            self.assertTrue(all(row['interpretation']=='update_direction_effectiveness_only' for row in result['rows']))


if __name__=='__main__':
    unittest.main()
