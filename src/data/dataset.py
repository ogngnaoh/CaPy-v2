"""CaPy dataset and dataloader construction (FR-4.1, FR-4.2)."""

import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CaPyDataset(Dataset):
    """Dataset returning aligned (mol, morph, expr) triplets (FR-4.1).

    Each __getitem__ returns a dict:
        {"mol": Tensor[2048], "morph": Tensor[morph_dim],
         "expr": Tensor[expr_dim], "metadata": {...}}

    SCARF augmentation is applied to morph and expr during training only.
    """

    def __init__(
        self,
        parquet_path: str,
        feature_columns_path: str,
        scarf_enabled: bool = False,
        scarf_corruption_rate: float = 0.4,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def build_dataloaders(config) -> tuple:
    """Build train/val/test DataLoaders from config (FR-4.2)."""
    raise NotImplementedError
