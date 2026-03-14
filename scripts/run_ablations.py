"""Ablation matrix execution harness (FR-9.1).

Runs training for each config × seed combination in the ablation matrix.
Collects results from checkpoints and writes to results/ablation_runs.jsonl.

Usage:
    python3 scripts/run_ablations.py --matrix core
    python3 scripts/run_ablations.py --matrix core --resume
    python3 scripts/run_ablations.py --matrix core --configs B4 --seeds 42 123 456
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_matrix(matrix_name: str) -> dict:
    """Load ablation matrix config from YAML."""
    matrix_path = Path(f"configs/ablation/{matrix_name}.yaml")
    if not matrix_path.exists():
        logger.error("Matrix config not found: %s", matrix_path)
        sys.exit(1)
    with open(matrix_path) as f:
        return yaml.safe_load(f)


def build_run_list(
    matrix: dict,
    config_filter: list[str] | None = None,
    seed_override: list[int] | None = None,
) -> list[dict]:
    """Build list of (config, seed) runs from matrix definition."""
    configs = matrix["configs"]
    if config_filter:
        configs = {k: v for k, v in configs.items() if k in config_filter}

    runs = []
    for config_id, config_def in configs.items():
        seeds = seed_override if seed_override else config_def["seeds"]
        model = config_def.get("model")
        for seed in seeds:
            runs.append(
                {
                    "config_id": config_id,
                    "name": config_def["name"],
                    "description": config_def.get("description", ""),
                    "model": model,
                    "seed": seed,
                }
            )
    return runs


def extract_metrics_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract metrics dict."""
    import torch

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    result = {
        "best_epoch": ckpt.get("epoch", -1),
        "best_metric": ckpt.get("best_metric", -1),
    }
    # Include full val metrics if saved
    if "metrics" in ckpt:
        for k, v in ckpt["metrics"].items():
            result[k] = v
    return result


def run_training(
    model: str, seed: int, checkpoint_dir: str
) -> subprocess.CompletedProcess:
    """Execute train.py for a single config × seed."""
    cmd = [
        sys.executable,
        "scripts/train.py",
        f"model={model}",
        f"seed={seed}",
        f"checkpoint_dir={checkpoint_dir}",
    ]
    return subprocess.run(cmd, check=True)


def main() -> None:
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
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Run only specific configs (e.g., --configs B4 B5)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seeds (e.g., --seeds 42 123 456)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results output directory",
    )
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    runs = build_run_list(matrix, args.configs, args.seeds)
    total = len(runs)
    logger.info("Ablation matrix: %d runs (%s)", total, args.matrix)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "ablation_runs.jsonl"

    completed = 0
    skipped = 0
    failed = 0

    for i, run in enumerate(runs, 1):
        config_id = run["config_id"]
        seed = run["seed"]
        model = run["model"]
        name = run["name"]

        # Baselines (B0-B3) don't train — they use raw features
        if model is None:
            logger.info(
                "Run %d/%d: config=%s, seed=%d. Status: baseline (skip)",
                i,
                total,
                config_id,
                seed,
            )
            skipped += 1
            continue

        checkpoint_path = checkpoint_dir / f"{name}_seed{seed}.pt"

        # Check for existing checkpoint
        if args.resume and checkpoint_path.exists():
            logger.info(
                "Run %d/%d: config=%s, seed=%d. Status: skipped (exists)",
                i,
                total,
                config_id,
                seed,
            )
            # Still collect metrics from existing checkpoint
            try:
                metrics = extract_metrics_from_checkpoint(checkpoint_path)
                run_result = {
                    "config": config_id,
                    "name": name,
                    "model": model,
                    "seed": seed,
                    **metrics,
                }
                with open(results_path, "a") as f:
                    f.write(json.dumps(run_result) + "\n")
            except Exception as e:
                logger.warning("Could not extract metrics: %s", e)
            skipped += 1
            continue

        logger.info(
            "Run %d/%d: config=%s, seed=%d. Status: running",
            i,
            total,
            config_id,
            seed,
        )

        try:
            run_training(model, seed, args.checkpoint_dir)
            completed += 1
            logger.info(
                "Run %d/%d: config=%s, seed=%d. Status: complete",
                i,
                total,
                config_id,
                seed,
            )

            # Collect metrics from checkpoint
            if checkpoint_path.exists():
                metrics = extract_metrics_from_checkpoint(checkpoint_path)
                run_result = {
                    "config": config_id,
                    "name": name,
                    "model": model,
                    "seed": seed,
                    **metrics,
                }
                with open(results_path, "a") as f:
                    f.write(json.dumps(run_result) + "\n")
                logger.info(
                    "  -> mean_R@10=%.4f at epoch %d",
                    metrics.get("best_metric", -1),
                    metrics.get("best_epoch", -1),
                )

        except subprocess.CalledProcessError as e:
            failed += 1
            logger.error(
                "Run %d/%d: config=%s, seed=%d. Status: FAILED (rc=%d)",
                i,
                total,
                config_id,
                seed,
                e.returncode,
            )

    logger.info(
        "Ablation complete: %d completed, %d skipped, %d failed (of %d)",
        completed,
        skipped,
        failed,
        total,
    )


if __name__ == "__main__":
    main()
