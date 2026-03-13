"""Loss functions: SigLIP + VICReg (FR-6.1, FR-6.2, FR-6.3)."""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as f

_MODALITY_ORDER = {"mol": 0, "morph": 1, "expr": 2}


class SigLIPLoss(nn.Module):
    """SigLIP pairwise contrastive loss (FR-6.1).

    Computes: mean(-log_sigmoid(targets * sim + bias))
    where targets[i,j] = +1 if i==j, -1 otherwise.
    Bias is a learnable parameter initialized to bias_init.
    """

    def __init__(self, bias_init: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute SigLIP loss between two sets of embeddings.

        Args:
            z_a: Embeddings of shape [N, 256], L2-normalized.
            z_b: Embeddings of shape [N, 256], L2-normalized.

        Returns:
            Scalar loss value.
        """
        sim = z_a @ z_b.T  # [N, N] cosine similarity (inputs are L2-normed)
        n = z_a.shape[0]
        targets = 2 * torch.eye(n, device=z_a.device) - 1  # +1 diag, -1 off-diag
        logits = targets * sim + self.bias
        return -f.logsigmoid(logits).mean()


class VICRegLoss(nn.Module):
    """VICReg regularization loss (FR-6.2).

    Variance term: hinge loss on per-feature std (target std >= 1).
    Covariance term: penalizes off-diagonal covariance entries.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute VICReg loss for one modality's embeddings.

        Args:
            z: Embeddings of shape [N, 256].

        Returns:
            Scalar loss (variance_loss + covariance_loss).
        """
        n, d = z.shape

        # Variance: hinge on per-feature std
        std = torch.sqrt(z.var(dim=0) + self.eps)
        variance_loss = f.relu(1.0 - std).mean()

        # Covariance: penalize off-diagonal entries
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d

        return variance_loss + cov_loss


def compute_total_loss(
    embeddings: dict[str, torch.Tensor],
    siglib_loss_fn: SigLIPLoss,
    vicreg_loss_fn: VICRegLoss,
    vicreg_lambda: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss for all active modality pairs (FR-6.3).

    Args:
        embeddings: Dict of L2-normalized embeddings per active modality.
        siglib_loss_fn: SigLIP loss function instance.
        vicreg_loss_fn: VICReg loss function instance.
        vicreg_lambda: Weight for VICReg regularization terms.

    Returns:
        (total_loss, loss_dict) where loss_dict has individual components.
    """
    loss_dict: dict[str, float] = {}
    device = next(iter(embeddings.values())).device
    total_loss = torch.tensor(0.0, device=device)

    # Sort modalities by canonical order: mol, morph, expr
    modalities = sorted(embeddings.keys(), key=lambda m: _MODALITY_ORDER[m])

    # SigLIP for every pair of active modalities
    for m_a, m_b in itertools.combinations(modalities, 2):
        pair_loss = siglib_loss_fn(embeddings[m_a], embeddings[m_b])
        loss_dict[f"loss_{m_a}_{m_b}"] = pair_loss.item()
        total_loss = total_loss + pair_loss

    # VICReg per active modality (scaled by lambda)
    for m in modalities:
        vreg_loss = vicreg_loss_fn(embeddings[m])
        scaled = vicreg_lambda * vreg_loss
        loss_dict[f"vicreg_{m}"] = scaled.item()
        total_loss = total_loss + scaled

    loss_dict["loss_total"] = total_loss.item()
    return total_loss, loss_dict
