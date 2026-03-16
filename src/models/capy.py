"""CaPy model assembly (FR-5.5).

Assembles encoders + projection heads based on config.
Only instantiates components for active modalities.
"""

import torch
import torch.nn as nn

from src.models.encoders import (
    ExpressionEncoder,
    MolecularEncoder,
    MorphologyEncoder,
)
from src.models.projections import ProjectionHead
from src.utils.logging import get_logger

logger = get_logger(__name__)

_ENCODER_CLASSES = {
    "mol": MolecularEncoder,
    "morph": MorphologyEncoder,
    "expr": ExpressionEncoder,
}


class CaPyModel(nn.Module):
    """Tri-modal contrastive model with configurable modality selection (FR-5.5).

    For config T1 (modalities: [mol, morph, expr]): all 3 encoders + heads.
    For config B4 (modalities: [mol, morph]): only mol and morph.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.modalities = list(config.model.modalities)

        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        proj_cfg = config.model.projection

        for m in self.modalities:
            enc_cfg = config.model[f"{m}_encoder"]
            self.encoders[m] = _ENCODER_CLASSES[m](
                input_dim=enc_cfg.input_dim,
                output_dim=enc_cfg.output_dim,
                dropout=enc_cfg.dropout,
                hidden_dims=list(enc_cfg.hidden_dims),
            )
            self.projections[m] = ProjectionHead(
                input_dim=proj_cfg.input_dim,
                hidden_dim=proj_cfg.hidden_dim,
                output_dim=proj_cfg.output_dim,
            )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CaPyModel: modalities=%s, %d total params",
            self.modalities,
            total_params,
        )
        for m in self.modalities:
            enc_params = sum(p.numel() for p in self.encoders[m].parameters())
            proj_params = sum(p.numel() for p in self.projections[m].parameters())
            logger.info(
                "  %s: encoder=%d params, projection=%d params",
                m,
                enc_params,
                proj_params,
            )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass returning embeddings and encoder outputs.

        Args:
            batch: Dict with keys "mol", "morph", "expr" (whichever are active).

        Returns:
            (embeddings, encoder_outputs) where embeddings are L2-normalized
            256-dim vectors and encoder_outputs are pre-projection 512-dim vectors.
        """
        embeddings = {}
        encoder_outputs = {}
        for m in self.modalities:
            h = self.encoders[m](batch[m])
            z = self.projections[m](h)
            embeddings[m] = z
            encoder_outputs[m] = h
        return embeddings, encoder_outputs
