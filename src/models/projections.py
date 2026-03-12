"""Projection heads for contrastive learning (FR-5.4)."""

import torch
import torch.nn as nn
import torch.nn.functional as f


class ProjectionHead(nn.Module):
    """2-layer MLP projection head with L2 normalization (FR-5.4).

    Architecture: Linear(512→512) → BN → ReLU → Linear(512→256) → L2-normalize.
    Output: unit-norm vectors in the 256-dim contrastive space.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize. Output shape: [batch_size, output_dim]."""
        x = self.net(x)
        return f.normalize(x, p=2, dim=-1)
