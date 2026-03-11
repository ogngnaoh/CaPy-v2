"""MOA clustering evaluation (FR-8.3)."""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_moa_clustering(
    embeddings: torch.Tensor,
    moa_labels: list[str],
    k_values: list[int] = [5, 10, 20],
) -> dict[str, float]:
    """Compute MOA clustering metrics (FR-8.3).

    Returns dict with keys: "AMI", "ARI", "kNN_5_acc", "kNN_10_acc", "kNN_20_acc".
    Skips and warns if only 1 unique MOA label is found.
    """
    raise NotImplementedError
