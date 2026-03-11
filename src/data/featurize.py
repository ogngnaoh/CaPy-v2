"""Molecular featurization module (FR-3.1).

Converts SMILES strings to ECFP fingerprint tensors.
"""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def smiles_to_ecfp(
    smiles: str, n_bits: int = 2048, radius: int = 2
) -> torch.Tensor | None:
    """Convert a SMILES string to an ECFP fingerprint tensor (FR-3.1).

    Args:
        smiles: SMILES string for a compound.
        n_bits: Number of bits in the fingerprint (default: 2048).
        radius: Morgan fingerprint radius (default: 2).

    Returns:
        Float32 tensor of shape [n_bits], or None if SMILES parsing fails.
        If parsing fails, strips CXSMILES annotations and retries.
    """
    raise NotImplementedError
