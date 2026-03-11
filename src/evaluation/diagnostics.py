"""Embedding quality diagnostics: alignment and uniformity (FR-8.2)."""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_alignment(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Compute alignment between positive pairs (FR-8.2).

    Alignment = mean(||z_a[i] - z_b[i]||^2). Lower is better.
    """
    raise NotImplementedError


def compute_uniformity(z: torch.Tensor) -> float:
    """Compute uniformity of embeddings (FR-8.2).

    Uniformity = log(mean(exp(-2 * ||z[i] - z[j]||^2))) for all i≠j.
    More negative is better. Values > -0.5 indicate collapse.
    """
    raise NotImplementedError
