"""Deep Q-Network on top of the GNN embedding (paper Sec. 2.3)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    """Eq. (5): Q(s, .) = W3 ReLU(W2 ReLU(W1 s + b1) + b2) + b3, with s = [v_G, f_C] (Eq. 3)."""

    def __init__(self, embed_dim: int, feature_dim: int, n_actions: int = 100):
        super().__init__()
        # v_G is an unbounded output of the GNN whose scale grows as the GNN trains;
        # normalising it keeps it comparable to f_C, whose entries lie in [0, 1]
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, v_g: torch.Tensor, f_c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.norm(v_g), f_c], dim=-1))

    @torch.no_grad()
    def greedy_action(self, v_g: torch.Tensor, f_c: np.ndarray, mask: np.ndarray | None = None) -> int:
        q = self(v_g.unsqueeze(0), torch.from_numpy(f_c).unsqueeze(0)).squeeze(0)
        if mask is not None:
            q = q.masked_fill(~torch.from_numpy(mask), -torch.inf)
        return int(q.argmax())


class Transitions(NamedTuple):
    graph_id: torch.Tensor
    f_c: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_f_c: torch.Tensor
    done: torch.Tensor
    next_mask: torch.Tensor


class ReplayBuffer:
    """Uniform experience replay.

    The material is stored as a graph index, so that v_G can be looked up from
    the current GNN when a batch is replayed.
    """

    def __init__(self, capacity: int, feature_dim: int, n_actions: int, seed: int | None = None):
        self.capacity = capacity
        self.graph_id = np.zeros(capacity, dtype=np.int64)
        self.f_c = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_f_c = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.ones((capacity, n_actions), dtype=bool)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, graph_id: int, f_c: np.ndarray, action: int, reward: float, next_f_c: np.ndarray, done: bool,
             next_mask: np.ndarray | None = None):
        i = self.pos
        self.graph_id[i] = graph_id
        self.f_c[i] = f_c
        self.action[i] = action
        self.reward[i] = reward
        self.next_f_c[i] = next_f_c
        self.done[i] = float(done)
        self.next_mask[i] = True if next_mask is None else next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transitions:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return Transitions(*(torch.from_numpy(a[idx]) for a in (
            self.graph_id, self.f_c, self.action, self.reward, self.next_f_c, self.done, self.next_mask)))


def td_loss(q_net: QNetwork, target_net: QNetwork, v: torch.Tensor, batch: Transitions, gamma: float,
            double_dqn: bool = False) -> torch.Tensor:
    """Eq. (6): E[(r + gamma * max_a' Q(s', a'; theta^-) - Q(s, a; theta))^2].

    `v` is the material embedding v_G of each transition; when it carries
    gradients, the loss also trains the GNN. `batch.next_mask` is all True
    unless invalid-action masking is used. With `double_dqn` the online
    network selects a' and the target network evaluates it (van Hasselt et al.).
    """
    q = q_net(v, batch.f_c).gather(1, batch.action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(v, batch.next_f_c).masked_fill(~batch.next_mask, -torch.inf)
        if double_dqn:
            best = q_net(v, batch.next_f_c).masked_fill(~batch.next_mask, -torch.inf).argmax(dim=1, keepdim=True)
            next_q = next_q.gather(1, best).squeeze(1)
        else:
            next_q = next_q.max(dim=1).values
        # terminal states have no feasible action; zero them explicitly instead of 0 * -inf = nan
        next_q = torch.where(batch.done.bool(), torch.zeros_like(next_q), next_q)
        y = batch.reward + gamma * next_q
    return F.mse_loss(q, y)
