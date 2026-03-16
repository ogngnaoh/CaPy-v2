"""Entry point for data preprocessing (make preprocess).

Usage:
    python scripts/preprocess.py
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.data.preprocess import run_preprocessing_pipeline
from src.utils.logging import get_logger, setup_log_level

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess LINCS data")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    verbosity.add_argument("--quiet", action="store_true", help="Suppress info logging")
    args = parser.parse_args()

    if args.verbose:
        setup_log_level(logging.DEBUG)
    elif args.quiet:
        setup_log_level(logging.WARNING)

    cfg = OmegaConf.load(
        Path(__file__).parent.parent / "configs" / "data" / "lincs.yaml"
    )
    logger.info("Starting preprocessing pipeline")
    run_preprocessing_pipeline(config=cfg)
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
