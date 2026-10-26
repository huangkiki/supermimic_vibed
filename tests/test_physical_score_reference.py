from copy import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_policy_fidelity import fixture
from test_policy_estimators import ConstantMean
from experiments.ablation.policy_estimators import collect_score_rollout,pathwise_gradient,score_gradient,stochastic_probe


class ActionDynamics(nn.Module):
    def __init__(self,gain):
        super().__init__();self.gain=gain

    def step(self,state,action,xi):
        return tuple(self.gain*action for _ in state)


class PhysicalReferenceTests(unittest.TestCase):
    def test_control_variate_expectation_is_physical_gradient_despite_wrong_model(self):
        nodes,weights=np.polynomial.hermite.hermgauss(3)
        original,_=fixture(1,no_critic=True)
        original.policy=ConstantMean();original.batch=3
        original.starts={k:v[:1].expand(3,*v.shape[1:]).clone() for k,v in original.starts.items()}
        original.history={k:v[:1].expand(3,*v.shape[1:]).clone() for k,v in original.history.items()}
        original.starts['a_prev']=original.starts['a_prev'][:,:1]
        original.history['a_prev']=original.history['a_prev'][:,:,:1]
        original.mdp.add_noise=lambda obs:obs
        original.mdp.soft_termination=lambda state,frames:torch.zeros_like(state[0][...,0])
        quadrature=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float64)
        original.mdp.reward=lambda state,frames,action,previous:-(state[0][:,0]-.7).square()*quadrature*3
        original.wm=ActionDynamics(.5)
        probe=stochastic_probe(original,.15)
        physical=ActionDynamics(1.3)
        noise=torch.tensor(math.sqrt(2)*nodes,dtype=torch.float64).unsqueeze(-1)
        with patch('torch.randn_like',return_value=noise):
            physical_score=score_gradient(probe.policy,collect_score_rollout(probe,dynamics=physical))['gradient']
            model_score=score_gradient(probe.policy,collect_score_rollout(probe))['gradient']
            pathwise=pathwise_gradient(probe)['gradient']
        corrected=physical_score-model_score+pathwise
        analytic=2*1.3*(.7-1.3*.2)
        self.assertAlmostEqual(float(corrected),analytic,places=12)
        self.assertNotAlmostEqual(float(pathwise),analytic,places=3)
        self.assertAlmostEqual(float(model_score-pathwise),0.,places=12)


if __name__=='__main__':unittest.main()

class ComparisonBootstrapTests(unittest.TestCase):
    def test_simultaneous_interval_preserves_shared_noise_rows_and_location(self):
        from experiments.ablation.analyze_score_reference import simultaneous_intervals
        samples=np.random.default_rng(12).normal(size=(64,7))
        lower,upper,critical=simultaneous_intervals(samples,draws=2000)
        self.assertGreater(critical,1.96)
        self.assertTrue((lower<samples.mean(0)).all() and (upper>samples.mean(0)).all())
        lo,hi,again=simultaneous_intervals(samples+3,draws=2000)
        np.testing.assert_allclose(lo,lower+3,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(hi,upper+3,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(critical,again)
