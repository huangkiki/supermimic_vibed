import math
import unittest

import torch
from torch import nn

from experiments.ablation.calibrated_updates import calibrate_direction,CalibrationResolutionError
from experiments.ablation.policy_estimators import FixedGaussianPolicy
from experiments.ablation.policy_fidelity import parameter_vector


class ConstantMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor([.2,-.1],dtype=torch.float64))

    def forward(self,obs,goal):
        return self.bias.expand(len(obs),2)


class CalibrationTests(unittest.TestCase):
    def test_fp32_unresolvable_scale_retains_measurements_and_restores_weights(self):
        actor=FixedGaussianPolicy(ConstantMean().float(),.1)
        with torch.no_grad():actor.mean_policy.bias.fill_(10000.)
        initial=parameter_vector(actor).clone();inputs=torch.ones(2,2,1,1)
        direction=torch.ones(2)/math.sqrt(2)
        with self.assertRaises(CalibrationResolutionError) as raised:
            calibrate_direction(actor,inputs,inputs,direction,.0003,lambda:None)
        measured=raised.exception.measurements
        self.assertEqual(measured['requested_action_rms'],.0003)
        self.assertTrue(abs(measured['actual_action_rms']-.0003)>.000003 or measured['parameter_rounding_relative_l2']>.02)
        torch.testing.assert_close(parameter_vector(actor),initial,rtol=0,atol=0)

    def test_exact_gaussian_kl_and_step_are_recovered_without_mutation(self):
        actor=FixedGaussianPolicy(ConstantMean(),.1)
        initial=parameter_vector(actor).clone()
        inputs=torch.ones(4,3,1,1,dtype=torch.float64)
        direction=torch.ones(2,dtype=torch.float64)/math.sqrt(2)
        for target in (.0003,.001,.003):
            result=calibrate_direction(actor,inputs,inputs,direction,target,lambda:None)
            self.assertAlmostEqual(result['parameter_step']/(target*math.sqrt(2)),1,delta=.01)
            self.assertAlmostEqual(result['gaussian_kl']/(target**2/.1**2),1,delta=.021)
            torch.testing.assert_close(parameter_vector(actor),initial,rtol=0,atol=0)

    def test_unreachable_target_and_budget_failure_restore_parameters(self):
        actor=FixedGaussianPolicy(ConstantMean(),.1)
        initial=parameter_vector(actor).clone();inputs=torch.ones(2,2,1,1,dtype=torch.float64)
        direction=torch.ones(2,dtype=torch.float64)/math.sqrt(2)
        with self.assertRaisesRegex(ValueError,'cap'):
            calibrate_direction(actor,inputs,inputs,direction,.003,lambda:None,max_step=.0001)
        def fail():raise TimeoutError('exhausted')
        with self.assertRaises(TimeoutError):
            calibrate_direction(actor,inputs,inputs,direction,.003,fail)
        torch.testing.assert_close(parameter_vector(actor),initial,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
