"""MLP encoders for each modality (FR-5.1, FR-5.2, FR-5.3)."""

import torch
import torch.nn as nn


class MolecularEncoder(nn.Module):
    """ECFP → 512-dim encoder (FR-5.3).

    Architecture: Linear(2048→1024) → BN → ReLU → Dropout
                  → Linear(1024→1024) → BN → ReLU → Dropout
                  → Linear(1024→1024) → BN → ReLU → Dropout
                  → Linear(1024→512)
    """

    def __init__(self, input_dim: int = 2048, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MorphologyEncoder(nn.Module):
    """CellProfiler features → 512-dim encoder (FR-5.1).

    Architecture: Linear(morph_dim→1024) → BN → ReLU → Dropout
                  → Linear(1024→1024) → BN → ReLU → Dropout
                  → Linear(1024→512)
    """

    def __init__(self, input_dim: int = 1500, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ExpressionEncoder(nn.Module):
    """L1000 landmark genes → 512-dim encoder (FR-5.2).

    Architecture: identical to MorphologyEncoder with different input_dim.
    """

    def __init__(self, input_dim: int = 978, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
