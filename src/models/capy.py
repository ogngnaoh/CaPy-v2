"""CaPy model assembly (FR-5.5).

Assembles encoders + projection heads based on config.
Only instantiates components for active modalities.
"""

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CaPyModel(nn.Module):
    """Tri-modal contrastive model with configurable modality selection (FR-5.5).

    For config T1 (modalities: [mol, morph, expr]): all 3 encoders + heads.
    For config B4 (modalities: [mol, morph]): only mol and morph.
    """

    def __init__(self, config) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass returning embeddings for active modalities.

        Args:
            batch: Dict with keys "mol", "morph", "expr" (whichever are active).

        Returns:
            Dict of L2-normalized embeddings, e.g. {"mol": Tensor[N,256], ...}
        """
        raise NotImplementedError
