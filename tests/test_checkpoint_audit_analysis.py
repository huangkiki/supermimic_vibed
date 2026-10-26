from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.ablation.analyze import save_csv
from experiments.ablation.analyze_checkpoint_audit import analyze, batch_statistics


def fixture(root):
    plan = dict(wm_checkpoints=['/ckpt_001000.pt','/ckpt_004000.pt'], gradient_seeds=[1,2],
                reference_seeds=[3,4], direction_seed=5, epsilons=[.001,.003,.01],
                primary_epsilon=.003, directions=2, replicas=2, repeats=3)
    (root/'audit-plan.json').write_text(json.dumps(plan))
    (root/'audit-summary.json').write_text(json.dumps(dict(status='complete',parameter_count=10)))
    direction = (2*np.random.default_rng(5).integers(0,2,(2,3))-1)/np.sqrt(3)
    np.save(root/'bias-directions.npy',direction)
    methods=['ppo_sim', 'ckpt_001000_bptt','ckpt_001000_ppo','ckpt_004000_bptt','ckpt_004000_ppo']
    bias=np.array([.3,.2,.1]); gradients=[]; fd=[]; baselines=[]; prediction=[]; updates=[]
    for method in methods:
        for seed in plan['gradient_seeds']:
            gradients.append(dict(method=method,seed=seed,bias=bias.tolist(),projections=(direction@bias).tolist()))
        for seed in plan['reference_seeds']:
            for n in range(4):
                for epsilon in plan['epsilons']:
                    updates.append(dict(method=method,seed=seed,sample=n,epsilon=epsilon,
                        proxy_improvement=epsilon,hard_improvement=epsilon/2,contact_group='stable_contact'))
    for seed in plan['reference_seeds']:
        for n in range(4):
            baselines.append(dict(seed=seed,sample=n,proxy=1,hard=1,failed=False,contact_group='stable_contact'))
            for model in ('ckpt_001000','ckpt_004000'):
                for h in (1,8,16,32):
                    prediction.append(dict(checkpoint=model,seed=seed,sample=n,horizon=h,
                        scaled_mse=.01*h,contact_group='stable_contact'))
    for model in ('physical','ckpt_001000','ckpt_004000'):
        for d in range(2):
            for epsilon in plan['epsilons'] if model=='physical' else [plan['primary_epsilon']]:
                for sign in (1,-1):
                    for seed in plan['reference_seeds']:
                        for n in range(4):
                            fd.append(dict(model=model,direction=d,epsilon=epsilon,sign=sign,seed=seed,sample=n,
                                proxy=1+sign*epsilon*(direction[d]@bias),hard=1,relative_rounding=0))
    (root/'gradient-draws.json').write_text(json.dumps(gradients))
    (root/'baseline-repeats.json').write_text(json.dumps([dict(proxy=[1]*4,hard=[1]*4)]*2))
    for filename, data in [('directional-returns',fd),('physical-baselines',baselines),('prediction',prediction),('physical-updates',updates)]:
        save_csv(root/(filename+'.csv'),data)
    return fd


class AuditAnalysisTests(unittest.TestCase):
    def test_analytic_directional_reference_is_recovered_and_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            result=analyze(root,root/'analysis')
            self.assertEqual(result['validation']['resolved_by_epsilon'],{'0.001':2,'0.003':2,'0.01':2})
            self.assertLess(max(row['rmse_all'] for row in result['accuracy']),1e-12)
            self.assertEqual(result['validation']['independent_training_seeds'],1)

    def test_duplicate_and_missing_reference_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=fixture(root)
            data[1]=deepcopy(data[0]);save_csv(root/'directional-returns.csv',data)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                analyze(root,root/'analysis')

    def test_seed_clustering_does_not_treat_duplicated_worlds_as_independent_samples(self):
        records=[dict(method='a',seed=seed,value=value) for seed,value in [(1,1),(2,3)] for _ in range(100)]
        result=batch_statistics(records,('method',),'value')[0]
        self.assertEqual(result['mean'],2)
        self.assertEqual(result['standard_error'],1)
        self.assertEqual(result['noise_batches'],2)


if __name__ == '__main__':
    unittest.main()
