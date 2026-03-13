"""Cross-modal retrieval metrics (FR-8.1)."""

import itertools

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODALITY_ORDER = {"mol": 0, "morph": 1, "expr": 2}


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
    sim = z_a @ z_b.T  # [N, N] cosine similarity (inputs are L2-normed)

    # For each query i, rank of correct match i = number of candidates
    # with similarity >= the correct match's similarity
    correct_sim = sim.diag().unsqueeze(1)  # [N, 1]
    ranks = (sim >= correct_sim).sum(dim=1)  # [N], 1-indexed

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = (ranks <= k).float().mean().item()
    metrics["MRR"] = (1.0 / ranks.float()).mean().item()
    return metrics


def compute_all_retrieval_metrics(
    embeddings: dict[str, torch.Tensor], ks: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """Compute retrieval metrics for all active modality pairs (both directions).

    Returns dict with keys like "mol->morph/R@1", "morph->mol/R@10", "mean_R@10".
    """
    modalities = sorted(embeddings.keys(), key=lambda m: _MODALITY_ORDER[m])
    result: dict[str, float] = {}
    r10_values: list[float] = []

    for m_a, m_b in itertools.permutations(modalities, 2):
        metrics = compute_retrieval_metrics(embeddings[m_a], embeddings[m_b], ks)
        for key, val in metrics.items():
            result[f"{m_a}->{m_b}/{key}"] = val
        r10_values.append(metrics["R@10"])

    result["mean_R@10"] = sum(r10_values) / len(r10_values)
    return result
