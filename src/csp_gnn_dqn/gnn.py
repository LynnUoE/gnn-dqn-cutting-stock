"""GCN encoder with Jumping Knowledge and global mean pooling (paper Sec. 2.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool


class TextureGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 64, num_classes: int = 10):
        super().__init__()
        # Eq. (1): three GCN layers, H' = ReLU(D^-1/2 (A + I) D^-1/2 H W)
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)]
        )
        # Eq. (2): concatenate {h^(0), ..., h^(3)} followed by a linear transformation
        self.jk = JumpingKnowledge("cat")
        self.jk_linear = nn.Linear(in_dim + 3 * hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Graph-level vector v_G used as the material part of the DQN state."""
        hs = [x]
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            hs.append(h)
        h = self.jk_linear(self.jk(hs))
        return global_mean_pool(h, batch)

    def forward(self, data) -> torch.Tensor:
        return self.classifier(self.encode(data.x, data.edge_index, data.batch))
