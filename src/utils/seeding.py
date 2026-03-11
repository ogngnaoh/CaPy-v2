"""Reproducibility seeding (FR-10.2)."""

import random

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seed all random sources for reproducibility.

    Seeds: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Sets torch.backends.cudnn.deterministic = True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeded all random sources with seed=%d", seed)
