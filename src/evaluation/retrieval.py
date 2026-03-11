"""Cross-modal retrieval metrics (FR-8.1)."""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_retrieval_metrics(
    z_a: torch.Tensor, z_b: torch.Tensor, ks: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """Compute retrieval metrics for one direction (FR-8.1).

    Args:
        z_a: Query embeddings of shape [N, D].
        z_b: Candidate embeddings of shape [N, D].
        ks: List of K values for R@K.

    Returns:
        Dict with keys "R@1", "R@5", "R@10", "MRR".
    """
    raise NotImplementedError


def compute_all_retrieval_metrics(
    embeddings: dict[str, torch.Tensor], ks: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """Compute retrieval metrics for all active modality pairs (both directions).

    Returns dict with keys like "mol->morph/R@1", "morph->mol/R@10", "mean_R@10".
    """
    raise NotImplementedError
