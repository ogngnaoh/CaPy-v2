"""Projection heads for contrastive learning (FR-5.4)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP projection head with L2 normalization (FR-5.4).

    Architecture: Linear(512→512) → BN → ReLU → Linear(512→256) → L2-normalize.
    Output: unit-norm vectors in the 256-dim contrastive space.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize. Output shape: [batch_size, output_dim]."""
        raise NotImplementedError
