"""Entry point for data preprocessing (make preprocess).

Usage:
    python scripts/preprocess.py
"""

from src.data.preprocess import run_preprocessing_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting preprocessing pipeline")
    run_preprocessing_pipeline(config=None)  # TODO: load Hydra config
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
