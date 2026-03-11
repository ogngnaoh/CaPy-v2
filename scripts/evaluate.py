"""Entry point for model evaluation (make evaluate).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --full
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --diagnostics
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --clustering
"""

import argparse

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CaPy model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--full", action="store_true", help="Run full evaluation report")
    parser.add_argument("--diagnostics", action="store_true", help="Run alignment/uniformity diagnostics")
    parser.add_argument("--clustering", action="store_true", help="Run MOA clustering evaluation")
    args = parser.parse_args()

    logger.info("Starting evaluation")
    raise NotImplementedError("Evaluation entry point not yet implemented")


if __name__ == "__main__":
    main()
