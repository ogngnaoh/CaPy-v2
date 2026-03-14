"""Loss functions: SigLIP + VICReg (FR-6.1, FR-6.2, FR-6.3)."""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as f

_MODALITY_ORDER = {"mol": 0, "morph": 1, "expr": 2}


class SigLIPLoss(nn.Module):
    """SigLIP pairwise contrastive loss (FR-6.1).

    Computes: mean(-log_sigmoid(targets * (temperature * sim + bias)))
    where targets[i,j] = +1 for positive pairs, -1 otherwise.
    Temperature and bias are learnable parameters (following SigLIP paper).

    Supports compound-aware multi-positive pairing (OPEN-1): when compound_ids
    are provided, all samples from the same compound are treated as positive
    pairs, eliminating false negatives from dose-level duplication.
    """

    def __init__(
        self, bias_init: float = 0.0, log_temp_init: float = 2.0
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias_init))
        self.log_temperature = nn.Parameter(torch.tensor(log_temp_init))

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        compound_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute SigLIP loss between two sets of embeddings.

        Args:
            z_a: Embeddings of shape [N, 256], L2-normalized.
            z_b: Embeddings of shape [N, 256], L2-normalized.
            compound_ids: Optional list of compound IDs per sample. When
                provided, same-compound pairs are treated as positives.

        Returns:
            Scalar loss value.
        """
        sim = z_a @ z_b.T  # [N, N] cosine similarity (inputs are L2-normed)
        n = z_a.shape[0]

        if compound_ids is not None:
            # Multi-positive: same compound = positive pair (OPEN-1)
            unique_ids = {cid: i for i, cid in enumerate(set(compound_ids))}
            idx = torch.tensor(
                [unique_ids[cid] for cid in compound_ids], device=z_a.device
            )
            targets = 2 * (idx.unsqueeze(1) == idx.unsqueeze(0)).float() - 1
        else:
            targets = 2 * torch.eye(n, device=z_a.device) - 1  # +1 diag, -1 off-diag

        temperature = self.log_temperature.exp().clamp(min=1.0, max=30.0)
        logits = targets * (temperature * sim + self.bias)
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
    encoder_outputs: dict[str, torch.Tensor] | None = None,
    compound_ids: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss for all active modality pairs (FR-6.3).

    Args:
        embeddings: Dict of L2-normalized embeddings per active modality.
        siglib_loss_fn: SigLIP loss function instance.
        vicreg_loss_fn: VICReg loss function instance.
        vicreg_lambda: Weight for VICReg regularization terms.
        encoder_outputs: Dict of pre-projection encoder outputs (for VICReg).
            If None, falls back to embeddings (backward compat).
        compound_ids: Optional list of compound IDs for multi-positive pairing.

    Returns:
        (total_loss, loss_dict) where loss_dict has individual components.
    """
    loss_dict: dict[str, float] = {}
    device = next(iter(embeddings.values())).device
    total_loss = torch.tensor(0.0, device=device)

    # Sort modalities by canonical order: mol, morph, expr
    modalities = sorted(embeddings.keys(), key=lambda m: _MODALITY_ORDER[m])

    # SigLIP for every pair of active modalities (on L2-normalized embeddings)
    for m_a, m_b in itertools.combinations(modalities, 2):
        pair_loss = siglib_loss_fn(
            embeddings[m_a], embeddings[m_b], compound_ids
        )
        loss_dict[f"loss_{m_a}_{m_b}"] = pair_loss.item()
        total_loss = total_loss + pair_loss

    # VICReg per active modality (on pre-normalization encoder outputs)
    vicreg_inputs = encoder_outputs if encoder_outputs is not None else embeddings
    for m in modalities:
        vreg_loss = vicreg_loss_fn(vicreg_inputs[m])
        scaled = vicreg_lambda * vreg_loss
        loss_dict[f"vicreg_{m}"] = scaled.item()
        total_loss = total_loss + scaled

    loss_dict["loss_total"] = total_loss.item()
    return total_loss, loss_dict
