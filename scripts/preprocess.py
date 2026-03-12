"""Entry point for data preprocessing (make preprocess).

Usage:
    python scripts/preprocess.py
"""

from pathlib import Path

from omegaconf import OmegaConf

from src.data.preprocess import run_preprocessing_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    cfg = OmegaConf.load(Path(__file__).parent.parent / "configs" / "data" / "lincs.yaml")
    logger.info("Starting preprocessing pipeline")
    run_preprocessing_pipeline(config=cfg)
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
