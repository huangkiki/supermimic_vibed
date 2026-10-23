"""Likelihood-ratio PPO. Rollouts, targets, and old log probabilities are detached."""

from __future__ import annotations

import math

import torch
from torch import nn


class GaussianActor(nn.Module):
    """Preserve the original deterministic mean network; add 29 learned log stds.

    The sampled action is NOT clipped or squashed. Simulator actuator limits remain
    active. Clipping a Gaussian action while scoring the clipped value would give
    an incorrect likelihood ratio. Evaluation uses only ``mean_policy``.
    """

    def __init__(self, mean_policy: nn.Module, action_dim: int = 29, initial_std: float = 0.1):
        super().__init__()
        if initial_std <= 0:
            raise ValueError("initial_std must be positive")
        self.mean_policy = mean_policy
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(initial_std)))

    def distribution(self, observations, goals):
        mean = self.mean_policy(observations, goals)
        return torch.distributions.Normal(mean, self.log_std.clamp(-5, 1).exp())

    @torch.no_grad()
    def sample(self, observations, goals):
        dist = self.distribution(observations, goals)
        actions = dist.sample()
        return actions, dist.log_prob(actions).sum(-1)

    def score(self, observations, goals, actions):
        dist = self.distribution(observations, goals)
        return dist.log_prob(actions.detach()).sum(-1), dist.entropy().sum(-1)


@torch.no_grad()
def gae(rewards, values, next_values, bootstrap, trace, gamma=0.99, lam=0.95):
    """Separate terminal bootstrap from episode-boundary trace masking.

    ``bootstrap=0`` only for true terminals; timeouts bootstrap from their PRE-reset
    next state. ``trace=0`` at any reset, so GAE never leaks into a new episode.
    Imagined survival uses 1-d_soft for both masks, matching the BPTT objective.
    All inputs have shape (time, environments).
    """
    if not all(x.shape == rewards.shape for x in (values, next_values, bootstrap, trace)):
        raise ValueError("GAE inputs must have equal (time, batch) shapes")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * bootstrap[t] * next_values[t] - values[t]
        carry = delta + gamma * lam * trace[t] * carry
        advantages[t] = carry
    return advantages, advantages + values


def ppo_loss(log_prob, old_log_prob, advantage, weight, clip):
    ratio = (log_prob - old_log_prob.detach()).exp()
    advantage, weight = advantage.detach(), weight.detach()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -(weight * surrogate).mean()


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, rollout, *,
               epochs=5, minibatches=4, clip=0.2, entropy_coef=0.0, target_kl=0.02):
    """Reuse one on-policy batch; never recompute normalizers or targets in epochs."""
    data = {name: value.detach().flatten(0, 1) for name, value in rollout.items()}
    weights, advantages = data["weight"], data["advantage"]
    denom = weights.sum().clamp_min(1e-8)
    mean = (weights * advantages).sum() / denom
    var = (weights * (advantages - mean).square()).sum() / denom
    data["advantage"] = (advantages - mean) / (var.sqrt() + 1e-8)
    size = len(weights)
    actor_updates, critic_updates = 0, 0
    actor_stopped = False
    kls, policy_losses, value_losses = [], [], []
    for _ in range(epochs):
        for ids in torch.randperm(size, device=weights.device).tensor_split(minibatches):
            if not len(ids):
                continue
            if not actor_stopped:
                logp, entropy = actor.score(data["obs"][ids], data["goal"][ids], data["action"][ids])
                log_ratio = logp - data["log_prob"][ids]
                with torch.no_grad():
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                kls.append(float(kl))
                if target_kl > 0 and kl > target_kl:
                    actor_stopped = True
                else:
                    loss = ppo_loss(logp, data["log_prob"][ids], data["advantage"][ids], weights[ids], clip)
                    loss = loss - entropy_coef * (weights[ids] * entropy).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
                    actor_optimizer.step()
                    actor_updates += 1
                    policy_losses.append(float(loss.detach()))
            v_loss = (critic(data["critic_input"][ids]) - data["target"][ids]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            critic_optimizer.step()
            critic_updates += 1
            value_losses.append(float(v_loss.detach()))
    return {
        "policy_loss": sum(policy_losses) / max(len(policy_losses), 1),
        "value_loss": sum(value_losses) / max(len(value_losses), 1),
        "approx_kl": max(kls, default=0.0), "actor_updates": actor_updates,
        "critic_updates": critic_updates, "kl_stopped": actor_stopped,
    }
