"""Post-training diagnostic script.

Loads a checkpoint and generates diagnostic plots + statistics to identify
issues with alignment, collapse, overfitting, or parameter pathology.

Usage:
    python scripts/diagnose.py --checkpoint checkpoints/bi_mol_morph_seed42.pt
    python scripts/diagnose.py --checkpoint checkpoints/bi_mol_morph_seed42.pt --split val
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as f
from omegaconf import OmegaConf

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint and return contents."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def print_siglip_parameters(checkpoint: dict) -> None:
    """Print learned SigLIP temperature and bias."""
    siglip_state = checkpoint.get("siglip_state_dict", {})
    log_temp = siglip_state.get("log_temperature", None)
    bias = siglip_state.get("bias", None)

    print("\n=== SigLIP Learned Parameters ===")
    if log_temp is not None:
        temp = log_temp.exp().item()
        print(f"  log_temperature: {log_temp.item():.4f}")
        print(f"  temperature:     {temp:.4f}")
    else:
        print("  log_temperature: NOT FOUND in checkpoint")

    if bias is not None:
        print(f"  bias:            {bias.item():.4f}")
    else:
        print("  bias:            NOT FOUND in checkpoint")

    # Interpret
    if log_temp is not None and bias is not None:
        temp = log_temp.exp().item()
        b = bias.item()
        # Decision boundary: sim = -b/temp
        if temp > 0:
            boundary = -b / temp
            print(f"\n  Decision boundary (sim = -bias/temp): {boundary:.4f}")
            print(f"  Positive pair (sim=1.0) logit: {temp * 1.0 + b:.2f}")
            print(f"  Negative pair (sim=0.0) logit: {-(temp * 0.0 + b):.2f}")
            pos_loss = -f.logsigmoid(torch.tensor(temp * 1.0 + b)).item()
            neg_loss = -f.logsigmoid(torch.tensor(-(temp * 0.0 + b))).item()
            print(f"  Positive pair loss at sim=1: {pos_loss:.4f}")
            print(f"  Negative pair loss at sim=0: {neg_loss:.4f}")



def generate_diagnostics(
    checkpoint_path: str, split: str = "val", output_dir: str = "results/diagnostics"
) -> None:
    """Full diagnostic pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"Best metric: {checkpoint.get('best_metric', '?')}")

    # Print SigLIP parameters
    print_siglip_parameters(checkpoint)

    # Build model and dataset
    cfg = OmegaConf.create(config)
    modalities = list(cfg.model.modalities)

    # Override input dims
    feature_columns_path = cfg.data.output.feature_columns_path
    if Path(feature_columns_path).exists():
        with open(feature_columns_path) as fh:
            feature_cols = json.load(fh)
        from omegaconf import open_dict

        with open_dict(cfg):
            dim_map = {
                "morph_encoder": len(feature_cols["morph_features"]),
                "expr_encoder": len(feature_cols["expr_features"]),
            }
            for enc_key, actual_dim in dim_map.items():
                mod = enc_key.split("_")[0]
                if mod in cfg.model.modalities:
                    cfg.model[enc_key].input_dim = actual_dim

    from src.data.dataset import CaPyDataset, collate_fn
    from src.models.capy import CaPyModel

    model = CaPyModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build dataset
    processed_dir = Path(cfg.data.output.processed_dir)
    ds = CaPyDataset(
        parquet_path=str(processed_dir / f"{split}.parquet"),
        feature_columns_path=feature_columns_path,
        scarf_enabled=False,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )

    # Collect embeddings
    all_emb: dict[str, list] = {m: [] for m in modalities}
    with torch.no_grad():
        for batch in loader:
            batch_input = {
                k: v for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            embeddings, _ = model(batch_input)
            for m in modalities:
                all_emb[m].append(embeddings[m])

    emb = {m: torch.cat(tensors) for m, tensors in all_emb.items()}
    n_samples = emb[modalities[0]].shape[0]
    print(f"\n=== {split.upper()} Set: {n_samples} samples ===")
    print(f"Random R@10 baseline: {10/n_samples:.4f} ({10/n_samples*100:.2f}%)")

    # Cosine similarity analysis for each pair
    import itertools

    for m_a, m_b in itertools.combinations(modalities, 2):
        sim = emb[m_a] @ emb[m_b].T  # [N, N]
        pos_sims = sim.diag().numpy()  # matched pairs
        # Sample negatives (off-diagonal)
        mask = ~torch.eye(n_samples, dtype=torch.bool)
        neg_sims = sim[mask].numpy()
        neg_sample = np.random.choice(neg_sims, size=min(10000, len(neg_sims)), replace=False)

        print(f"\n--- {m_a} <-> {m_b} ---")
        print(f"  Positive cosine sim: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")
        print(f"  Negative cosine sim: mean={neg_sample.mean():.4f}, std={neg_sample.std():.4f}")
        print(f"  Separation (pos_mean - neg_mean): {pos_sims.mean() - neg_sample.mean():.4f}")
        alignment = (emb[m_a] - emb[m_b]).pow(2).sum(dim=1).mean().item()
        print(f"  Alignment (lower=better, random=2.0): {alignment:.4f}")

        # Plot cosine similarity distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(neg_sample, bins=80, alpha=0.6, label="Negative pairs", density=True, color="#4878cf")
        ax.hist(pos_sims, bins=40, alpha=0.7, label="Positive pairs (matched)", density=True, color="#e34a33")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"{m_a} <-> {m_b}: Cosine Similarity Distribution")
        ax.legend()
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Positive similarity histogram detail
        ax = axes[1]
        ax.hist(pos_sims, bins=40, alpha=0.8, color="#e34a33")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Positive Pair Similarities (mean={pos_sims.mean():.4f})")
        ax.axvline(x=pos_sims.mean(), color="black", linestyle="--", label=f"mean={pos_sims.mean():.3f}")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        fig_path = output_path / f"sim_dist_{m_a}_{m_b}_{split}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig_path}")

    # UMAP visualization
    try:
        import umap

        print("\n=== UMAP Embedding Visualization ===")

        # Stack all modality embeddings
        all_vectors = []
        all_labels = []
        for m in modalities:
            all_vectors.append(emb[m].numpy())
            all_labels.extend([m] * n_samples)

        all_vectors = np.vstack(all_vectors)
        all_labels = np.array(all_labels)

        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        umap_emb = reducer.fit_transform(all_vectors)

        # Plot colored by modality
        fig, ax = plt.subplots(figsize=(10, 8))
        palette = {"mol": "#e34a33", "morph": "#4878cf", "expr": "#6a994e"}
        for m in modalities:
            mask = all_labels == m
            ax.scatter(
                umap_emb[mask, 0],
                umap_emb[mask, 1],
                label=m,
                alpha=0.4,
                s=10,
                c=palette.get(m, "gray"),
            )
        ax.set_title(f"UMAP of {split} Embeddings (colored by modality)")
        ax.legend(markerscale=3)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        fig_path = output_path / f"umap_modality_{split}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved: {fig_path}")

        # Plot showing matched pairs (lines connecting mol<->morph for subset)
        if len(modalities) >= 2:
            m_a, m_b = modalities[0], modalities[1]
            n_show = min(50, n_samples)
            idx_a = np.where(all_labels == m_a)[0][:n_show]
            idx_b = np.where(all_labels == m_b)[0][:n_show]

            fig, ax = plt.subplots(figsize=(10, 8))
            for m in modalities:
                mask = all_labels == m
                ax.scatter(
                    umap_emb[mask, 0],
                    umap_emb[mask, 1],
                    label=m,
                    alpha=0.2,
                    s=8,
                    c=palette.get(m, "gray"),
                )
            # Draw lines connecting matched pairs
            for i in range(n_show):
                ax.plot(
                    [umap_emb[idx_a[i], 0], umap_emb[idx_b[i], 0]],
                    [umap_emb[idx_a[i], 1], umap_emb[idx_b[i], 1]],
                    "k-",
                    alpha=0.15,
                    linewidth=0.5,
                )
            ax.set_title(f"Matched Pairs: {m_a} <-> {m_b} (first {n_show})")
            ax.legend(markerscale=3)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

            fig_path = output_path / f"umap_pairs_{m_a}_{m_b}_{split}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Saved: {fig_path}")

    except ImportError:
        print("umap-learn not installed, skipping UMAP visualization")

    # Per-modality uniformity
    print("\n=== Per-Modality Uniformity ===")
    for m in modalities:
        sq_dists = torch.cdist(emb[m], emb[m]).pow(2)
        n = emb[m].shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        uniformity = torch.log(torch.exp(-2 * sq_dists[mask]).mean()).item()
        print(f"  {m}: uniformity = {uniformity:.4f} (>-0.5 = collapse)")

    # Summary and recommendations
    print("\n=== Summary ===")
    for m_a, m_b in itertools.combinations(modalities, 2):
        sim = emb[m_a] @ emb[m_b].T
        pos_mean = sim.diag().mean().item()
        neg_mean = sim[~torch.eye(n_samples, dtype=torch.bool)].mean().item()
        if pos_mean < neg_mean + 0.05:
            print(f"  WARNING: {m_a}<->{m_b} positive sims ({pos_mean:.3f}) "
                  f"not meaningfully above negative sims ({neg_mean:.3f})")
            print(f"           → Model is NOT learning cross-modal alignment")
        elif pos_mean < 0.3:
            print(f"  WEAK: {m_a}<->{m_b} some separation but weak "
                  f"(pos={pos_mean:.3f}, neg={neg_mean:.3f})")
        else:
            print(f"  OK: {m_a}<->{m_b} reasonable separation "
                  f"(pos={pos_mean:.3f}, neg={neg_mean:.3f})")

    print(f"\nDiagnostic plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-training diagnostics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostics",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    generate_diagnostics(args.checkpoint, args.split, args.output_dir)


if __name__ == "__main__":
    main()
