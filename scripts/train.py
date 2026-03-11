"""Entry point for model training (make train).

Usage:
    python scripts/train.py
    python scripts/train.py model=bi_mol_morph training.batch_size=128 seed=42
"""

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    # TODO: Hydra @hydra.main decorator with config_path="../configs", config_name="default"
    logger.info("Starting training")
    raise NotImplementedError("Training entry point not yet implemented")


if __name__ == "__main__":
    main()
