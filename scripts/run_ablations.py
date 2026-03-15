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

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data.featurize import featurize_smiles_batch
from src.evaluation.retrieval import compute_all_compound_retrieval_metrics
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

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


def load_existing_configs(results_path: Path) -> set[str]:
    """Load config IDs already present in the JSONL file."""
    existing = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing.add(entry.get("config", ""))
    return existing


def _load_val_data(
    processed_dir: Path,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load val split and feature column names.

    Returns:
        (df, morph_cols, expr_cols, compound_ids)
    """
    df = pd.read_parquet(processed_dir / "val.parquet")
    with open(processed_dir / "feature_columns.json") as fh:
        feature_cols = json.load(fh)
    morph_cols = feature_cols["morph_features"]
    expr_cols = feature_cols["expr_features"]

    # Deduplicate to compound level
    df = df.drop_duplicates(subset="compound_id").reset_index(drop=True)
    compound_ids = df["compound_id"].tolist()
    return df, morph_cols, expr_cols, compound_ids


def _build_raw_embeddings(
    df: pd.DataFrame,
    morph_cols: list[str],
    expr_cols: list[str],
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build 256-dim L2-normalized embeddings from raw features.

    Raw features have different dimensions per modality (ECFP=2048,
    morph~1500, expr=978). To enable cross-modal retrieval, each modality
    is projected to 256-dim via a deterministic random linear projection
    (preserves pairwise distances by Johnson-Lindenstrauss lemma).
    """
    embed_dim = 256

    # Molecular: ECFP fingerprints
    ecfp_dict = featurize_smiles_batch(df["smiles"].tolist())
    mol_raw = torch.stack([ecfp_dict[s] for s in df["smiles"]])

    # Morphology: CellProfiler features
    morph_raw = torch.tensor(df[morph_cols].fillna(0.0).values, dtype=torch.float32)

    # Expression: L1000 features
    expr_raw = torch.tensor(df[expr_cols].fillna(0.0).values, dtype=torch.float32)

    # Random linear projection to common 256-dim space (seeded for reproducibility)
    rng = torch.Generator()
    rng.manual_seed(seed)
    proj_mol = torch.randn(mol_raw.shape[1], embed_dim, generator=rng) / (
        mol_raw.shape[1] ** 0.5
    )
    proj_morph = torch.randn(morph_raw.shape[1], embed_dim, generator=rng) / (
        morph_raw.shape[1] ** 0.5
    )
    proj_expr = torch.randn(expr_raw.shape[1], embed_dim, generator=rng) / (
        expr_raw.shape[1] ** 0.5
    )

    mol = F.normalize(mol_raw @ proj_mol, dim=1)
    morph = F.normalize(morph_raw @ proj_morph, dim=1)
    expr = F.normalize(expr_raw @ proj_expr, dim=1)

    return {"mol": mol, "morph": morph, "expr": expr}


def evaluate_baseline(
    config_id: str,
    seed: int,
    processed_dir: Path,
) -> dict:
    """Evaluate B0-B3 baselines without training.

    B0: Random 256-dim L2-normalized embeddings for all 3 modalities.
    B1-B3: Raw features for all 3 modalities, but mean_R@10 averages
           only the 4 directions involving the baseline's modality.

    Returns:
        Dict with config, seed, and all compound retrieval metrics.
    """
    df, morph_cols, expr_cols, compound_ids = _load_val_data(processed_dir)
    n = len(df)
    logger.info("Evaluating baseline %s with %d compounds", config_id, n)

    if config_id == "B0":
        # Random embeddings (seeded for reproducibility)
        seed_everything(seed)
        embeddings = {
            "mol": F.normalize(torch.randn(n, 256), dim=1),
            "morph": F.normalize(torch.randn(n, 256), dim=1),
            "expr": F.normalize(torch.randn(n, 256), dim=1),
        }
        metrics = compute_all_compound_retrieval_metrics(embeddings, compound_ids)
        return {"config": config_id, "seed": seed, **metrics}

    # B1-B3: use raw features for all 3 modalities
    embeddings = _build_raw_embeddings(df, morph_cols, expr_cols)
    all_metrics = compute_all_compound_retrieval_metrics(embeddings, compound_ids)

    # Compute modality-specific mean_R@10 (average of 4 directions involving this modality)
    modality_map = {"B1": "mol", "B2": "morph", "B3": "expr"}
    primary_mod = modality_map[config_id]
    involved_directions = [
        k
        for k in all_metrics
        if k.startswith("compound/")
        and k.endswith("/R@10")
        and primary_mod in k.split("/")[1]
        and "->" in k
    ]
    if involved_directions:
        modality_mean = sum(all_metrics[d] for d in involved_directions) / len(
            involved_directions
        )
        all_metrics["compound/mean_R@10"] = modality_mean
        logger.info(
            "  %s mean_R@10 (over %d directions): %.4f",
            config_id,
            len(involved_directions),
            modality_mean,
        )

    return {"config": config_id, "seed": seed, **all_metrics}


def extract_metrics_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract metrics dict."""
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
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed data directory (for baseline evaluation)",
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

    # Load existing config IDs for --resume support
    existing_configs = load_existing_configs(results_path) if args.resume else set()

    completed = 0
    skipped = 0
    failed = 0

    for i, run in enumerate(runs, 1):
        config_id = run["config_id"]
        seed = run["seed"]
        model = run["model"]
        name = run["name"]

        # Baselines (B0-B3) don't train — evaluate raw features / random embeddings
        if model is None:
            # Resume: skip if already evaluated
            if args.resume and config_id in existing_configs:
                logger.info(
                    "Run %d/%d: config=%s, seed=%d. Status: skipped (baseline exists)",
                    i,
                    total,
                    config_id,
                    seed,
                )
                skipped += 1
                continue

            logger.info(
                "Run %d/%d: config=%s, seed=%d. Status: evaluating baseline",
                i,
                total,
                config_id,
                seed,
            )
            try:
                processed_dir = Path(args.processed_dir)
                baseline_result = evaluate_baseline(config_id, seed, processed_dir)
                baseline_result["name"] = name
                baseline_result["model"] = None
                with open(results_path, "a") as f_out:
                    f_out.write(json.dumps(baseline_result) + "\n")
                completed += 1
                logger.info(
                    "  -> %s compound/mean_R@10=%.4f",
                    config_id,
                    baseline_result.get("compound/mean_R@10", -1),
                )
            except Exception as e:
                failed += 1
                logger.error("Baseline %s failed: %s", config_id, e)
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
