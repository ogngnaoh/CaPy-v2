"""Ablation matrix execution harness (FR-9.1).

Usage:
    python scripts/run_ablations.py --matrix core
    python scripts/run_ablations.py --matrix core --resume
"""

import argparse

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ablation matrix")
    parser.add_argument(
        "--matrix",
        type=str,
        default="core",
        help="Ablation matrix name (matches configs/ablation/{name}.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs with existing checkpoints",
    )
    args = parser.parse_args()

    logger.info("Running ablation matrix: %s", args.matrix)
    raise NotImplementedError("Ablation harness not yet implemented")


if __name__ == "__main__":
    main()
