"""Hydra configuration utilities (FR-10.1)."""

import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_git_hash() -> str | None:
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def save_config_yaml(config: DictConfig, path: Path) -> None:
    """Save resolved Hydra config to a YAML file alongside checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.info("Config saved: %s", path)
