"""Loss functions: SigLIP + VICReg (FR-6.1, FR-6.2, FR-6.3)."""

import torch
import torch.nn as nn


class SigLIPLoss(nn.Module):
    """SigLIP pairwise contrastive loss (FR-6.1).

    Computes: mean(-log_sigmoid(targets * sim + bias))
    where targets[i,j] = +1 if i==j, -1 otherwise.
    Bias is a learnable parameter initialized to 0.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute SigLIP loss between two sets of embeddings.

        Args:
            z_a: Embeddings of shape [N, 256].
            z_b: Embeddings of shape [N, 256].

        Returns:
            Scalar loss value.
        """
        raise NotImplementedError


class VICRegLoss(nn.Module):
    """VICReg regularization loss (FR-6.2).

    Variance term: hinge loss on per-feature std (target std >= 1).
    Covariance term: penalizes off-diagonal covariance entries.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute VICReg loss for one modality's embeddings.

        Args:
            z: Embeddings of shape [N, 256].

        Returns:
            Scalar loss (variance_loss + covariance_loss).
        """
        raise NotImplementedError


def compute_total_loss(
    embeddings: dict[str, torch.Tensor],
    siglib_loss_fn: SigLIPLoss,
    vicreg_loss_fn: VICRegLoss,
    vicreg_lambda: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss for all active modality pairs (FR-6.3).

    Returns:
        (total_loss, loss_dict) where loss_dict has individual components.
    """
    raise NotImplementedError
