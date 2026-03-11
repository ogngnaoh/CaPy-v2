"""Hydra configuration utilities (FR-10.1)."""

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/default.yaml"):
    """Load and resolve Hydra configuration."""
    raise NotImplementedError


def log_config_to_wandb(config) -> None:
    """Log resolved config to W&B run."""
    raise NotImplementedError
