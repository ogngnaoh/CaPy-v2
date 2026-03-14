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
        feature_cols = json.load(fh)
    morph_cols = feature_cols["morph_features"]

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

    # Load expression features
    expr_cols = feature_cols["expr_features"]
    expr_df = df[expr_cols].fillna(0.0)
    expr = torch.tensor(expr_df.values, dtype=torch.float32)

    # L2-normalize all modalities
    mol = f.normalize(mol, dim=1)
    morph = f.normalize(morph, dim=1)
    expr = f.normalize(expr, dim=1)
    n = mol.shape[0]
    logger.info("Computing retrieval for %d compounds", n)

    random_baseline = {k: min(k, n) / n for k in [1, 5, 10]}

    # Compute retrieval for all 3 pairs (both directions)
    pairs = [
        ("mol", mol, "morph", morph),
        ("mol", mol, "expr", expr),
        ("morph", morph, "expr", expr),
    ]
    for name_a, z_a, name_b, z_b in pairs:
        for query_name, query, cand_name, cand in [
            (name_a, z_a, name_b, z_b),
            (name_b, z_b, name_a, z_a),
        ]:
            sim = query @ cand.T
            correct_sim = sim.diag().unsqueeze(1)
            ranks = (sim >= correct_sim).sum(dim=1)

            logger.info("--- Raw-feature retrieval (%s->%s) ---", query_name, cand_name)
            for k in [1, 5, 10]:
                recall = (ranks <= k).float().mean().item()
                logger.info(
                    "  R@%d: %.4f  (random: %.4f)", k, recall, random_baseline[k]
                )
            mrr = (1.0 / ranks.float()).mean().item()
            logger.info("  MRR: %.4f", mrr)

            r10 = (ranks <= 10).float().mean().item()
            verdict = "PASS" if r10 > random_baseline[10] * 2 else "WEAK"
            logger.info(
                "%s: %s->%s R@10=%.4f (random=%.4f)",
                verdict,
                query_name,
                cand_name,
                r10,
                random_baseline[10],
            )


if __name__ == "__main__":
    main()
