import unittest

import torch
from torch import nn

from experiments.ablation.ppo import GaussianActor, gae, ppo_loss, ppo_update


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, obs, goal):
        return self.layer(torch.cat((obs.flatten(-2), goal), -1))


class PPOTests(unittest.TestCase):
    def test_terminal_and_timeout_do_not_leak_across_resets(self):
        r = torch.tensor([[1., 1.], [100., 100.]])
        v = torch.zeros_like(r)
        nv = torch.tensor([[7., 7.], [0., 0.]])
        boot = torch.tensor([[0., 1.], [0., 0.]])
        trace = torch.zeros_like(r)
        adv, target = gae(r, v, nv, boot, trace, gamma=0.9)
        torch.testing.assert_close(target[0], torch.tensor([1., 7.3]))
        self.assertFalse(adv.requires_grad)

    def test_soft_survival_matches_lambda_return(self):
        r = torch.tensor([[2.], [4.]])
        v = torch.tensor([[1.], [3.]])
        nv = torch.tensor([[3.], [5.]])
        survival = torch.tensor([[0.5], [0.25]])
        _, target = gae(r, v, nv, survival, survival, gamma=0.9, lam=0.8)
        last = 4 + 0.9 * 0.25 * 5
        first = 2 + 0.9 * 0.5 * (0.2 * 3 + 0.8 * last)
        torch.testing.assert_close(target[:, 0], torch.tensor([first, last]))

    def test_same_policy_logprob_ratio_is_one(self):
        actor = GaussianActor(TinyPolicy(), 2)
        obs, goal = torch.randn(5, 1, 2), torch.randn(5, 1)
        a, old = actor.sample(obs, goal)
        new, _ = actor.score(obs, goal, a)
        torch.testing.assert_close((new - old).exp(), torch.ones(5))
        self.assertFalse(a.requires_grad)

    def test_clipping_stops_excess_positive_update(self):
        lp = torch.tensor([0.5], requires_grad=True)
        loss = ppo_loss(lp, torch.zeros(1), torch.ones(1), torch.ones(1), 0.2)
        loss.backward()
        self.assertEqual(lp.grad.item(), 0)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)

    def test_policy_update_cannot_backpropagate_into_world_model(self):
        torch.manual_seed(7)
        actor = GaussianActor(TinyPolicy(), 2)
        critic = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0))
        wm = nn.Linear(2, 2)
        obs = wm(torch.randn(4, 3, 2)).unsqueeze(-2)
        goal = torch.randn(4, 3, 1)
        actions, old = actor.sample(obs, goal)
        batch = dict(obs=obs, goal=goal, action=actions, log_prob=old,
                     critic_input=torch.randn(4, 3, 3), target=torch.randn(4, 3),
                     advantage=torch.randn(4, 3), weight=torch.ones(4, 3))
        before = actor.mean_policy.layer.weight.detach().clone()
        ppo_update(actor, critic, torch.optim.Adam(actor.parameters(), 1e-3),
                   torch.optim.Adam(critic.parameters(), 1e-3), batch, epochs=2, minibatches=2)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertFalse(torch.equal(before, actor.mean_policy.layer.weight))


if __name__ == "__main__":
    unittest.main()
