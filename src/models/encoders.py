"""MLP encoders for each modality (FR-5.1, FR-5.2, FR-5.3)."""

import torch
import torch.nn as nn


class _MLPEncoder(nn.Module):
    """Generic MLP encoder: [input] → (Linear→BN→ReLU→Dropout)×N → Linear → [output]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MolecularEncoder(nn.Module):
    """ECFP → encoder (FR-5.3).

    Architecture: [2048→512→256→output_dim] with BN, ReLU, Dropout.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 256,
        dropout: float = 0.3,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.net = _MLPEncoder(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MorphologyEncoder(nn.Module):
    """CellProfiler features → encoder (FR-5.1).

    Architecture: [~1500→512→256→output_dim] with BN, ReLU, Dropout.
    """

    def __init__(
        self,
        input_dim: int = 1500,
        output_dim: int = 256,
        dropout: float = 0.3,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.net = _MLPEncoder(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpressionEncoder(nn.Module):
    """L1000 landmark genes → encoder (FR-5.2).

    Architecture: [978→512→256→output_dim] with BN, ReLU, Dropout.
    """

    def __init__(
        self,
        input_dim: int = 978,
        output_dim: int = 256,
        dropout: float = 0.3,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.net = _MLPEncoder(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
