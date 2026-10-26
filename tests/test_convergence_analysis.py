"""Raw-data reanalysis detects corruption and independently checks full means."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_policy_fidelity import fixture
from experiments.ablation.analyze_estimator_convergence import paired_decomposition, validate
from experiments.ablation.common import sha256, write_json
from experiments.ablation.estimator_convergence import run_convergence
from experiments.ablation.policy_estimators import stochastic_probe


def make_case(path):
    probe = stochastic_probe(fixture(1)[0], .1)
    def physical():
        result = probe.objective()
        return dict(proxy=result['proxy'], hard=result['proxy'], initial_state_drift=torch.zeros(probe.batch))
    run_convergence(probe, physical, path, gradient_seeds=[1,2,3,4], reference_seeds=[11,12,13,14],
                    counts=[2,4], epsilons=[.001,.003], directions=3, direction_seed=42, repeats=3)
    torch.save(dict(test_fixture=True), path / 'policy-bank.pt')
    write_json(path / 'probe.json', dict(status='complete', prepared_bank_sha256=sha256(path / 'policy-bank.pt'),
        precision=dict(tf32=False,amp='None',compiled=False),
        args=dict(estimator_seeds=[1,2,3,4], reference_seeds=[11,12,13,14], convergence_counts=[2,4],
                  epsilons=[.001,.003], directions=3, direction_seed=42, gradient_horizon=1, action_std=.1,
                  repeats=3,sample_indices=list(range(probe.batch)))))


class ConvergenceAnalysisTests(unittest.TestCase):
    def test_paired_uncertainty_cancels_shared_noise(self):
        # The estimators/references each vary, but their paired gaps are constant.
        base = np.arange(4, dtype=float)
        projected = np.stack([base[:,None],base[:,None]+2.])
        physical = np.zeros((1,2,2,4,1))
        for index, epsilon in enumerate((.01,.03)):
            physical[0,index,0,:,0] = 2*epsilon*base
        wm = physical.copy()
        for index, epsilon in enumerate((.01,.03)):
            wm[0,index,0,:,0] += 2*epsilon*3.
        rows = paired_decomposition(dict(projections=projected,physical_proxy=physical,wm=wm,
                                         epsilons=np.array([.01,.03]),counts=np.array([2,4])))
        for row in rows:
            self.assertAlmostEqual(row['ppo_minus_bptt'],2.)
            self.assertAlmostEqual(row['wm_minus_physics'],3.)
            self.assertAlmostEqual(row['ppo_minus_bptt_se'],0.)
            self.assertAlmostEqual(row['wm_minus_physics_se'],0.)

    def test_raw_and_full_vector_checks_pass_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            make_case(path)
            raw,summary,costs,variance=validate(path,means_path=path/'final-means.pt')
            self.assertEqual(raw['projections'].shape,(2,4,3))
            self.assertEqual(len(variance),4)
            self.assertEqual(len(costs),4)
            self.assertEqual(summary['counts'],[2,4])
            self.assertFalse(torch.cuda.is_initialized())

    def test_modified_raw_returns_are_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            make_case(path)
            raw=dict(np.load(path/'raw-reference.npz'))
            raw['physical_proxy'][0,0,0,0,0]+=1.
            np.savez(path/'raw-reference.npz',**raw)
            with self.assertRaisesRegex(ValueError,'Raw reference hash'):
                validate(path)

    def test_full_vector_projection_check_detects_forged_mean_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            make_case(path)
            means=torch.load(path/'final-means.pt',weights_only=True)
            means['bptt'].neg_()  # Same norm and variance, wrong directional estimates.
            torch.save(means,path/'final-means.pt')
            summary=json.loads((path/'convergence-summary.json').read_text())
            summary['means_sha256']=sha256(path/'final-means.pt')
            write_json(path/'convergence-summary.json',summary)
            with self.assertRaises(AssertionError):
                validate(path,means_path=path/'final-means.pt')


if __name__ == '__main__':
    unittest.main()
