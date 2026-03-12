"""Molecular featurization module (FR-3.1).

Converts SMILES strings to ECFP fingerprint tensors.
"""

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _strip_cxsmiles(smiles: str) -> str:
    """Strip CXSMILES annotation (everything after ' |')."""
    if not isinstance(smiles, str):
        return smiles
    idx = smiles.find(" |")
    if idx >= 0:
        return smiles[:idx]
    return smiles


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
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        stripped = _strip_cxsmiles(smiles)
        mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        logger.warning("Could not parse SMILES: %s", smiles)
        return None

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(mol)
    return torch.tensor(fp, dtype=torch.float32)
