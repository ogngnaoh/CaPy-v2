"""Verify cross-modal signal in raw features before learned embeddings.

Computes cosine similarity and kNN retrieval (R@1/5/10) between ECFP
fingerprints and raw CellProfiler morphology features at compound level.
A non-trivial R@10 (above 1/N random baseline) confirms learnable signal.

Usage:
    python3 scripts/verify_signal.py
    python3 scripts/verify_signal.py --processed-dir data/processed --split val
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as f

from src.data.featurize import featurize_smiles_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cross-modal signal")
    parser.add_argument(
        "--processed-dir", default="data/processed", help="Processed data directory"
    )
    parser.add_argument("--split", default="val", help="Split to evaluate")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    df = pd.read_parquet(processed_dir / f"{args.split}.parquet")
    with open(processed_dir / "feature_columns.json") as fh:
        morph_cols = json.load(fh)["morph_features"]

    # Deduplicate to compound level (one row per compound)
    df = df.drop_duplicates(subset="compound_id").reset_index(drop=True)
    logger.info("Loaded %d unique compounds from %s split", len(df), args.split)

    # Featurize SMILES -> ECFP
    ecfp_dict = featurize_smiles_batch(df["smiles"].tolist())
    valid = df["smiles"].isin(ecfp_dict.keys())
    df = df[valid].reset_index(drop=True)
    mol = torch.stack([ecfp_dict[s] for s in df["smiles"]])
    morph_df = df[morph_cols].fillna(0.0)
    morph = torch.tensor(morph_df.values, dtype=torch.float32)

    # L2-normalize
    mol = f.normalize(mol, dim=1)
    morph = f.normalize(morph, dim=1)
    n = mol.shape[0]
    logger.info("Computing %d x %d similarity matrix", n, n)

    # Cosine similarity: matched vs random pairs
    sim = mol @ morph.T
    matched_sim = sim.diag().mean().item()
    off_diag = sim[~torch.eye(n, dtype=torch.bool)].mean().item()
    logger.info("Mean cosine sim — matched: %.4f, random: %.4f", matched_sim, off_diag)

    # kNN retrieval: R@1, R@5, R@10
    correct_sim = sim.diag().unsqueeze(1)
    ranks = (sim >= correct_sim).sum(dim=1)  # 1-indexed
    random_baseline = {k: min(k, n) / n for k in [1, 5, 10]}

    logger.info("--- Raw-feature retrieval (mol->morph) ---")
    for k in [1, 5, 10]:
        recall = (ranks <= k).float().mean().item()
        logger.info(
            "  R@%d: %.4f  (random baseline: %.4f)", k, recall, random_baseline[k]
        )

    mrr = (1.0 / ranks.float()).mean().item()
    logger.info("  MRR: %.4f", mrr)

    r10 = (ranks <= 10).float().mean().item()
    if r10 > random_baseline[10] * 2:
        logger.info(
            "PASS: raw-feature R@10 (%.4f) exceeds 2x random (%.4f)",
            r10,
            random_baseline[10],
        )
    else:
        logger.info(
            "WEAK: raw-feature R@10 (%.4f) near random (%.4f)", r10, random_baseline[10]
        )


if __name__ == "__main__":
    main()
