"""Cross-modal retrieval metrics (FR-8.1)."""

import itertools

import torch
import torch.nn.functional as f

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


def compute_compound_retrieval_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    compound_ids: list[str],
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Compute compound-level retrieval (one row per compound).

    Averages embeddings per compound, re-normalizes, then computes R@K
    on the deduplicated set. This eliminates rank inflation from dose
    duplication where identical (mol, morph) pairs appear multiple times.

    Args:
        z_a: Query embeddings of shape [N, D].
        z_b: Candidate embeddings of shape [N, D].
        compound_ids: List of compound IDs per sample (length N).
        ks: List of K values for R@K.

    Returns:
        Dict with keys "R@1", "R@5", "R@10", "MRR", "n_compounds".
    """
    unique_ids = list(dict.fromkeys(compound_ids))  # preserve order
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    n_compounds = len(unique_ids)
    d = z_a.shape[1]

    # Average embeddings per compound
    z_a_dedup = torch.zeros(n_compounds, d, device=z_a.device)
    z_b_dedup = torch.zeros(n_compounds, d, device=z_b.device)
    counts = torch.zeros(n_compounds, device=z_a.device)

    for i, cid in enumerate(compound_ids):
        idx = id_to_idx[cid]
        z_a_dedup[idx] += z_a[i]
        z_b_dedup[idx] += z_b[i]
        counts[idx] += 1

    z_a_dedup /= counts.unsqueeze(1)
    z_b_dedup /= counts.unsqueeze(1)

    # Re-normalize after averaging
    z_a_dedup = f.normalize(z_a_dedup, dim=-1)
    z_b_dedup = f.normalize(z_b_dedup, dim=-1)

    metrics = compute_retrieval_metrics(z_a_dedup, z_b_dedup, ks)
    metrics["n_compounds"] = float(n_compounds)
    return metrics


def compute_all_compound_retrieval_metrics(
    embeddings: dict[str, torch.Tensor],
    compound_ids: list[str],
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Compound-level retrieval for all modality pairs (both directions).

    Returns dict with keys like "compound/mol->morph/R@10", "compound/mean_R@10".
    """
    modalities = sorted(embeddings.keys(), key=lambda m: _MODALITY_ORDER[m])
    result: dict[str, float] = {}
    r10_values: list[float] = []

    for m_a, m_b in itertools.permutations(modalities, 2):
        metrics = compute_compound_retrieval_metrics(
            embeddings[m_a], embeddings[m_b], compound_ids, ks
        )
        for key, val in metrics.items():
            result[f"compound/{m_a}->{m_b}/{key}"] = val
        r10_values.append(metrics["R@10"])

    result["compound/mean_R@10"] = (
        sum(r10_values) / len(r10_values) if r10_values else 0.0
    )
    # Get n_compounds from compound_ids directly
    n_compounds = len(dict.fromkeys(compound_ids))
    result["compound/random_R@10"] = min(10.0 / max(n_compounds, 1), 1.0)
    logger.info(
        "Compound-level retrieval: %d unique compounds, random_R@10=%.4f",
        n_compounds,
        result["compound/random_R@10"],
    )
    return result


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
