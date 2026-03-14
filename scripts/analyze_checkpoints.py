"""Analyze and compare per-direction metrics from saved checkpoints.

Loads all checkpoints from a directory, extracts per-direction compound-level
R@10, and prints a comparison table.

Usage:
    python3 scripts/analyze_checkpoints.py
    python3 scripts/analyze_checkpoints.py --checkpoint-dir checkpoints
"""

import argparse
from pathlib import Path

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_metrics_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract metrics dict."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    result = {
        "best_epoch": ckpt.get("epoch", -1),
        "best_metric": ckpt.get("best_metric", -1),
    }
    if "metrics" in ckpt:
        for k, v in ckpt["metrics"].items():
            result[k] = v
    if "config" in ckpt:
        result["config_name"] = ckpt["config"].get("model", {}).get(
            "name", "unknown"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-direction metrics across checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        logger.error("Checkpoint directory not found: %s", ckpt_dir)
        return

    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        logger.info("No checkpoint files found in %s", ckpt_dir)
        return

    logger.info("Found %d checkpoint(s) in %s", len(ckpt_files), ckpt_dir)

    # Collect all direction keys across checkpoints
    all_directions = set()
    results = []
    for path in ckpt_files:
        metrics = extract_metrics_from_checkpoint(path)
        metrics["filename"] = path.name
        results.append(metrics)
        for key in metrics:
            if (
                key.startswith("compound/")
                and key.endswith("/R@10")
                and "->" in key
            ):
                all_directions.add(key)

    directions = sorted(all_directions)

    # Print header
    dir_labels = [d.split("/")[1] for d in directions]
    header = f"{'Checkpoint':<40} {'Epoch':>5} {'Mean R@10':>9}"
    for label in dir_labels:
        header += f" {label:>12}"
    logger.info(header)
    logger.info("-" * len(header))

    # Print rows
    for r in results:
        name = r.get("config_name", r["filename"])
        row = f"{name:<40} {r['best_epoch']:>5} {r['best_metric']:>9.4f}"
        for d in directions:
            val = r.get(d, float("nan"))
            row += f" {val:>12.4f}"
        logger.info(row)


if __name__ == "__main__":
    main()
