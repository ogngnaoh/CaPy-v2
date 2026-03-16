"""Entry point for model evaluation (make evaluate).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --full
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --diagnostics
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --clustering
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""

import argparse
import logging

from src.utils.logging import get_logger, setup_log_level

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CaPy model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation report",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run alignment/uniformity diagnostics",
    )
    parser.add_argument(
        "--clustering",
        action="store_true",
        help="Run MOA clustering evaluation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Path to output directory",
    )
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

    logger.info("Starting evaluation")

    if args.full:
        from src.evaluation.report import generate_full_report

        generate_full_report(args.checkpoint, args.data_dir, args.output_dir)
    else:
        import torch

        from src.evaluation.report import (
            compute_all_metrics,
            generate_embeddings,
            load_model_and_config,
            print_summary_table,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _config = load_model_and_config(args.checkpoint, device)

        # Build test loader
        from pathlib import Path

        from src.data.dataset import CaPyDataset, collate_fn

        data_dir = Path(args.data_dir)
        test_dataset = CaPyDataset(
            parquet_path=str(data_dir / "test.parquet"),
            feature_columns_path=str(data_dir / "feature_columns.json"),
            scarf_enabled=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        embeddings, compound_ids, moa_labels = generate_embeddings(
            model, test_loader, device
        )

        if args.diagnostics:
            import itertools

            from src.evaluation.diagnostics import (
                compute_alignment,
                compute_uniformity,
            )

            modalities = sorted(
                embeddings.keys(), key=lambda m: {"mol": 0, "morph": 1, "expr": 2}[m]
            )
            for m_a, m_b in itertools.combinations(modalities, 2):
                align = compute_alignment(embeddings[m_a], embeddings[m_b])
                logger.info("Alignment %s-%s: %.4f", m_a, m_b, align)
            for m in modalities:
                unif = compute_uniformity(embeddings[m])
                logger.info("Uniformity %s: %.4f", m, unif)

        elif args.clustering:
            from src.evaluation.clustering import compute_moa_clustering

            # Deduplicate to compound level using first modality
            first_mod = list(embeddings.keys())[0]
            unique_ids = list(dict.fromkeys(compound_ids))
            id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
            n_compounds = len(unique_ids)
            d = embeddings[first_mod].shape[1]

            compound_emb = torch.zeros(n_compounds, d)
            counts = torch.zeros(n_compounds)
            compound_moas: list[str | None] = [None] * n_compounds

            for i, cid in enumerate(compound_ids):
                idx = id_to_idx[cid]
                compound_emb[idx] += embeddings[first_mod][i]
                counts[idx] += 1
                if compound_moas[idx] is None and moa_labels[i] is not None:
                    compound_moas[idx] = moa_labels[i]

            compound_emb /= counts.unsqueeze(1)
            compound_emb = torch.nn.functional.normalize(compound_emb, dim=-1)

            clustering_metrics = compute_moa_clustering(compound_emb, compound_moas)
            for k, v in clustering_metrics.items():
                logger.info("MOA %s: %.4f", k, v)

        else:
            # Default: retrieval metrics only
            metrics = compute_all_metrics(embeddings, compound_ids, moa_labels)
            print_summary_table(metrics)


if __name__ == "__main__":
    main()
