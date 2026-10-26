"""Matched stochastic-target BPTT and on-policy PPO gradient diagnostics.

This auxiliary comparison changes the deterministic actor into a fixed-variance
Gaussian for BOTH estimators. It does not change either production trainer.
PPO is evaluated at theta_old: clipping is inactive, lambda=1, no advantage
normalization or entropy term. Gradients retain the units of the original J.
"""
from __future__ import annotations

from copy import copy
import math

import torch
from torch import nn

from .policy_fidelity import fixed_rng, preserve_requires_grad
from .ppo import ppo_loss


class FixedGaussianPolicy(nn.Module):
    """Same mean-network parameters; fixed sigma and reparameterized action noise."""

    def __init__(self, mean_policy, std):
        super().__init__()
        if not math.isfinite(std) or std <= 0:
            raise ValueError('The shared Gaussian standard deviation must be positive and finite')
        self.mean_policy = mean_policy
        self.std = float(std)
        self.records = None

    def forward(self, history, goal):
        mean = self.mean_policy(history, goal)
        action = mean + self.std * torch.randn_like(mean)
        if self.records is not None:
            self.records.append((history.detach().clone(), goal.detach().clone(),
                                 action.detach().clone(), self.log_prob(mean, action).detach().clone()))
        return action

    def log_prob(self, mean, actions):
        return torch.distributions.Normal(mean, self.std).log_prob(actions).sum(-1)


def stochastic_probe(probe, std):
    """Keep original snapshots untouched; only the auxiliary actor is stochastic."""
    if probe.detach_action_reward or isinstance(probe.policy, FixedGaussianPolicy):
        raise ValueError('Require a full-reward deterministic probe before matching estimators')
    result = copy(probe)
    result.policy = FixedGaussianPolicy(probe.policy, std)
    return result


def _gradient(objective, policy):
    values = torch.autograd.grad(objective, tuple(policy.parameters()))
    vector = torch.cat([value.detach().reshape(-1) for value in values])
    if not torch.isfinite(vector).all():
        raise FloatingPointError('Nonfinite policy-gradient estimate')
    return vector


def pathwise_gradient(probe):
    """Reparameterized BPTT through the unchanged original actor objective."""
    namespace = probe._namespace(probe.policy, probe.wm)
    with torch.enable_grad(), fixed_rng(probe.seed, probe.device), preserve_requires_grad(probe.wm, probe.critic):
        result = namespace['actor_objective']()
        gradient = _gradient(result['J'].mean(), probe.policy)
    return dict(gradient=gradient, objective=result['J'].detach().clone())


@torch.no_grad()
def collect_score_rollout(probe, *, dynamics=None):
    """Collect detached data through the SAME original actor prefix and noise tape."""
    actor = probe.policy
    if not isinstance(actor, FixedGaussianPolicy) or probe.detach_action_reward:
        raise ValueError('Score estimates require the shared stochastic full-reward target')
    if actor.records is not None:
        raise RuntimeError('A policy tape is already being collected')
    actor.records = []
    try:
        dynamics = probe.wm if dynamics is None else dynamics
        namespace = probe._namespace(actor, dynamics)
        with fixed_rng(probe.seed, probe.device), preserve_requires_grad(dynamics, probe.critic):
            result = namespace['actor_objective']()
        tape = actor.records
    finally:
        actor.records = None
    if len(tape) != probe.horizon:
        raise ValueError('Original actor prefix did not produce exactly H policy calls')
    # Values BEFORE each action are action-independent control variates.
    states = [result['S0'], *result['Ss']]
    frames = [probe.starts['f'], *result['Fs']]
    goals = [result['g0'], *result['Gs']]
    values = torch.stack([
        probe.critic(namespace['_critic_input'](state, frame, result['xi_n'], goal))
        for state, frame, goal in zip(states, frames, goals)
    ]) if not probe.no_critic else result['r'].new_zeros((probe.horizon + 1, probe.batch))
    tail = values[-1]
    targets = []
    for step in reversed(range(probe.horizon)):
        tail = result['r'][step] + probe.gamma * (1 - result['d'][step]) * tail
        targets.append(tail)
    targets = torch.stack(targets[::-1])
    torch.testing.assert_close(targets[0], result['J'], rtol=2e-6, atol=2e-7)
    observations, goal_inputs, actions, log_prob = (torch.stack(items) for items in zip(*tape))
    return dict(obs=observations, goal=goal_inputs, action=actions, old_log_prob=log_prob,
                advantage=(targets - values[:-1]).detach(), weight=(result['disc'] * result['c_t']).detach(),
                target=targets.detach(), value=values.detach(), objective=result['J'].detach().clone(),
                proxy=result['J'].detach().clone(), rewards=result['r'].detach().clone(),
                soft_termination=result['d'].detach().clone(), actions=actions,
                states=tuple(torch.stack([state[i] for state in states]).detach().clone()
                             for i in range(len(states[0]))),
                seed=probe.seed, std=actor.std, gamma=probe.gamma, no_critic=probe.no_critic,
                interpretation='on-policy PPO surrogate at theta_old; lambda=1; unnormalized advantages')


def score_gradient(actor, rollout, *, clip=.2):
    """PPO first-step gradient, with no path through stored observations or targets.

    The existing PPO loss averages H*B elements. Multiplying by H restores the
    original sum-over-time, mean-over-starts objective scale. No fitting of the
    critic, std, normalization or optimizer is performed here.
    """
    if not isinstance(actor, FixedGaussianPolicy) or rollout['std'] != actor.std:
        raise ValueError('The policy distribution must match the sampling distribution')
    history, goals, actions = (rollout[key].detach() for key in ('obs', 'goal', 'action'))
    with torch.enable_grad():
        # Preserve the original per-step batch shape. Flattening H*B selects
        # different CUDA GEMMs and can move log_prob despite identical weights.
        log_prob = torch.stack([actor.log_prob(actor.mean_policy(obs, goal), action)
                                for obs, goal, action in zip(history, goals, actions)])
        if not torch.equal(log_prob.detach(), rollout['old_log_prob']):
            error = float((log_prob.detach()-rollout['old_log_prob']).abs().max())
            raise ValueError(f'Log probability differs at theta_old (max error {error:g}); '
                             'require unchanged policy and identical numerical evaluation')
        loss = len(rollout['action']) * ppo_loss(log_prob, rollout['old_log_prob'],
            rollout['advantage'], rollout['weight'], clip)
        gradient = _gradient(-loss, actor)
    return dict(gradient=gradient, surrogate=-loss.detach())


def estimate_pair(probe):
    """Equal trajectories and target, different gradient estimators; no updates."""
    import time
    def synchronize():
        if probe.device.type == 'cuda':
            torch.cuda.synchronize(probe.device)
    synchronize()
    start = time.perf_counter()
    pathwise = pathwise_gradient(probe)
    synchronize()
    pathwise_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rollout = collect_score_rollout(probe)
    score = score_gradient(probe.policy, rollout)
    synchronize()
    score_seconds = time.perf_counter() - start
    torch.testing.assert_close(pathwise['objective'], rollout['objective'], rtol=0, atol=0)
    return dict(bptt=pathwise['gradient'], ppo=score['gradient'], rollout=rollout,
                bptt_seconds=pathwise_seconds, ppo_seconds=score_seconds,
                imagined_steps_per_estimator=probe.horizon * probe.batch)
