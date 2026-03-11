"""Data download module for LINCS cpg0004 dataset (FR-1.1, FR-1.2, FR-1.3)."""

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def download_morphology(target_dir: str = "data/raw/morphology/") -> Path:
    """Download Cell Painting profiles from S3 (FR-1.1).

    Downloads from s3://cellpainting-gallery/cpg0004-lincs/ using aws CLI
    with --no-sign-request. Falls back to HTTPS if aws CLI unavailable.
    Skips if file already exists with expected size.
    """
    raise NotImplementedError


def download_expression(target_dir: str = "data/raw/expression/") -> Path:
    """Download L1000 expression profiles from Figshare (FR-1.2)."""
    raise NotImplementedError


def download_metadata(target_dir: str = "data/raw/metadata/") -> Path:
    """Download compound metadata from Drug Repurposing Hub (FR-1.3)."""
    raise NotImplementedError


def download_all() -> None:
    """Download all three data sources."""
    download_morphology()
    download_expression()
    download_metadata()
    logger.info("All downloads complete.")
