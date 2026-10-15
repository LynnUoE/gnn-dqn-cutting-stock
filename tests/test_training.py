import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from csp_gnn_dqn import plots
from csp_gnn_dqn.dqn import QNetwork, ReplayBuffer, td_loss


class TDLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q_net = QNetwork(embed_dim=4, feature_dim=3, n_actions=5)
        self.buffer = ReplayBuffer(16, feature_dim=3, n_actions=5, seed=0)

    def test_terminal_state_without_feasible_action_is_finite(self):
        no_action = np.zeros(5, dtype=bool)
        for i in range(8):
            self.buffer.push(0, np.ones(3, np.float32), i % 5, 0.1, np.zeros(3, np.float32), True, no_action)
        batch = self.buffer.sample(8)
        for double in (False, True):
            loss = td_loss(self.q_net, self.q_net, torch.zeros(8, 4), batch, 0.99, double)
            self.assertTrue(torch.isfinite(loss))

    def test_terminal_target_is_the_reward(self):
        self.buffer.push(0, np.ones(3, np.float32), 2, 0.25, np.zeros(3, np.float32), True, np.zeros(5, bool))
        batch = self.buffer.sample(1)
        v = torch.zeros(1, 4)
        q = self.q_net(v, batch.f_c)[0, 2]
        loss = td_loss(self.q_net, self.q_net, v, batch, 0.99)
        self.assertAlmostEqual(loss.item(), (q.item() - 0.25) ** 2, places=6)

    def test_gradient_reaches_the_embedding(self):
        self.buffer.push(0, np.ones(3, np.float32), 1, 0.1, np.ones(3, np.float32), False)
        v = torch.zeros(1, 4, requires_grad=True)
        td_loss(self.q_net, self.q_net, v, self.buffer.sample(1), 0.99).backward()
        self.assertIsNotNone(v.grad)


class RewardPlotTest(unittest.TestCase):
    def test_any_run_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (5, 95, 100, 105, 125, 40000):
                plots.reward_and_epsilon(np.zeros(n), np.ones(n), Path(tmp) / f"fig6_{n}.png")


if __name__ == "__main__":
    unittest.main()
