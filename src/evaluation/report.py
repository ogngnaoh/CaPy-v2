"""Full evaluation report generation (FR-8.4)."""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import umap
from omegaconf import OmegaConf

from src.evaluation.clustering import compute_moa_clustering
from src.evaluation.diagnostics import compute_alignment, compute_uniformity
from src.evaluation.retrieval import (
    compute_all_compound_retrieval_metrics,
    compute_all_retrieval_metrics,
)
from src.models.capy import CaPyModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODALITY_ORDER = {"mol": 0, "morph": 1, "expr": 2}


def load_model_and_config(
    checkpoint_path: str, device: str | None = None
) -> tuple[CaPyModel, dict]:
    """Load model and config from a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load model onto. Defaults to CPU.

    Returns:
        (model, config) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = OmegaConf.create(checkpoint["config"])
    model = CaPyModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model from %s (epoch %d, best_metric=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("best_metric", 0.0),
    )
    return model, config


def generate_embeddings(
    model: CaPyModel,
    data_loader,
    device: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[str], list[str | None]]:
    """Generate embeddings for all data in the loader.

    Args:
        model: Trained CaPyModel in eval mode.
        data_loader: DataLoader yielding batches with tensors + metadata.
        device: Device to run inference on.

    Returns:
        (embeddings_dict, compound_ids, moa_labels) where embeddings_dict
        maps modality name to [N, D] tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_embeddings: dict[str, list[torch.Tensor]] = {
        m: [] for m in model.modalities
    }
    all_compound_ids: list[str] = []
    all_moa_labels: list[str | None] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch_gpu = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            embeddings, _ = model(batch_gpu)

            for m in model.modalities:
                all_embeddings[m].append(embeddings[m].cpu())

            if "metadata" in batch:
                for meta in batch["metadata"]:
                    all_compound_ids.append(meta["compound_id"])
                    all_moa_labels.append(meta.get("moa"))

    concat_embeddings = {
        m: torch.cat(tensors) for m, tensors in all_embeddings.items()
    }
    return concat_embeddings, all_compound_ids, all_moa_labels


def compute_all_metrics(
    embeddings: dict[str, torch.Tensor],
    compound_ids: list[str],
    moa_labels: list[str | None],
) -> dict[str, float]:
    """Compute all evaluation metrics: retrieval, diagnostics, and MOA clustering.

    Args:
        embeddings: Dict mapping modality name to [N, D] tensor.
        compound_ids: List of compound IDs (length N).
        moa_labels: List of MOA labels (length N), may contain None.

    Returns:
        Merged dict of all metrics.
    """
    metrics: dict[str, float] = {}

    # Row-level retrieval (FR-8.1)
    metrics.update(compute_all_retrieval_metrics(embeddings))

    # Compound-level retrieval (FR-8.1)
    if compound_ids:
        metrics.update(
            compute_all_compound_retrieval_metrics(embeddings, compound_ids)
        )

    # Alignment + uniformity (FR-8.2)
    modalities = sorted(
        embeddings.keys(), key=lambda m: _MODALITY_ORDER[m]
    )
    for m_a, m_b in itertools.combinations(modalities, 2):
        metrics[f"align_{m_a}_{m_b}"] = compute_alignment(
            embeddings[m_a], embeddings[m_b]
        )
    for m in modalities:
        metrics[f"uniform_{m}"] = compute_uniformity(embeddings[m])

    # MOA clustering (FR-8.3) — deduplicate to compound level
    if compound_ids and moa_labels:
        unique_ids = list(dict.fromkeys(compound_ids))
        id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
        n_compounds = len(unique_ids)
        d = next(iter(embeddings.values())).shape[1]

        # Average embeddings across modalities per compound
        all_emb = torch.zeros(n_compounds, d)
        counts = torch.zeros(n_compounds)
        for m in modalities:
            for i, cid in enumerate(compound_ids):
                idx = id_to_idx[cid]
                all_emb[idx] += embeddings[m][i]
                if m == modalities[0]:
                    counts[idx] += 1
        # Average over samples (not modalities) — use first modality for dedup
        first_mod = modalities[0]
        compound_emb = torch.zeros(n_compounds, d)
        for i, cid in enumerate(compound_ids):
            idx = id_to_idx[cid]
            compound_emb[idx] += embeddings[first_mod][i]
        compound_emb /= counts.unsqueeze(1)
        compound_emb = f.normalize(compound_emb, dim=-1)

        # Map compound_ids to MOA labels (take first non-null per compound)
        compound_moas: list[str | None] = [None] * n_compounds
        for i, cid in enumerate(compound_ids):
            idx = id_to_idx[cid]
            if compound_moas[idx] is None and moa_labels[i] is not None:
                compound_moas[idx] = moa_labels[i]

        clustering_metrics = compute_moa_clustering(
            compound_emb, compound_moas
        )
        for k, v in clustering_metrics.items():
            metrics[f"moa/{k}"] = v

    return metrics


def generate_retrieval_table(
    metrics: dict[str, float],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate retrieval results as CSV and LaTeX tables.

    Args:
        metrics: Full metrics dict from compute_all_metrics.
        output_dir: Directory to save output files.

    Returns:
        (csv_path, tex_path) tuple.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Collect all directions
    directions = set()
    for key in metrics:
        if "->" in key and "/" in key:
            parts = key.split("/")
            for part in parts:
                if "->" in part:
                    directions.add(part)

    for direction in sorted(directions):
        row = {"direction": direction}
        # Row-level metrics
        for metric_name in ["R@1", "R@5", "R@10", "MRR"]:
            key = f"{direction}/{metric_name}"
            row[f"row_{metric_name}"] = metrics.get(key, float("nan"))
        # Compound-level metrics
        for metric_name in ["R@1", "R@5", "R@10", "MRR"]:
            key = f"compound/{direction}/{metric_name}"
            row[f"compound_{metric_name}"] = metrics.get(
                key, float("nan")
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "retrieval_table.csv"
    tex_path = output_dir / "retrieval_table.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")

    logger.info("Retrieval table saved to %s and %s", csv_path, tex_path)
    return csv_path, tex_path


def generate_umap_plots(
    embeddings: dict[str, torch.Tensor],
    moa_labels: list[str | None],
    output_dir: Path,
) -> list[Path]:
    """Generate UMAP plots per modality, colored by MOA.

    Args:
        embeddings: Dict mapping modality name to [N, D] tensor.
        moa_labels: List of MOA labels (length N), may contain None.
        output_dir: Directory to save output files.

    Returns:
        List of saved PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Assign colors to MOA labels
    valid_moas = [m for m in moa_labels if m is not None]
    unique_moas = sorted(set(valid_moas))
    cmap = plt.get_cmap("tab20")
    moa_to_color = {
        moa: cmap(i % 20) for i, moa in enumerate(unique_moas)
    }
    grey = (0.7, 0.7, 0.7, 0.5)

    for modality in sorted(
        embeddings.keys(), key=lambda m: _MODALITY_ORDER[m]
    ):
        emb_np = embeddings[modality].detach().cpu().numpy()

        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(emb_np)

        colors = [
            moa_to_color.get(m, grey) if m is not None else grey
            for m in moa_labels
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.7
        )
        ax.set_title(f"UMAP — {modality}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        path = output_dir / f"umap_{modality}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        logger.info("UMAP plot saved: %s", path)

    return paths


def generate_similarity_heatmap(
    embeddings: dict[str, torch.Tensor],
    output_dir: Path,
    max_samples: int = 50,
) -> Path:
    """Generate cross-modal cosine similarity heatmaps.

    Args:
        embeddings: Dict mapping modality name to [N, D] tensor.
        output_dir: Directory to save output file.
        max_samples: Maximum number of samples to include (for readability).

    Returns:
        Path to saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = sorted(
        embeddings.keys(), key=lambda m: _MODALITY_ORDER[m]
    )
    pairs = list(itertools.combinations(modalities, 2))
    n_pairs = len(pairs)

    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (m_a, m_b) in enumerate(pairs):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]

        z_a = embeddings[m_a][:max_samples]
        z_b = embeddings[m_b][:max_samples]
        sim = (z_a @ z_b.T).detach().cpu().numpy()

        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{m_a} vs {m_b}")
        ax.set_xlabel(m_b)
        ax.set_ylabel(m_a)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_pairs, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Cross-modal Cosine Similarity", fontsize=14)
    fig.tight_layout()

    path = output_dir / "similarity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Similarity heatmap saved: %s", path)
    return path


def generate_training_curves(
    checkpoint_path: str,
    output_dir: Path,
) -> Path:
    """Generate training curves from checkpoint epoch history.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_dir: Directory to save output file.

    Returns:
        Path to saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    epoch_history = checkpoint.get("epoch_history", [])

    if not epoch_history:
        # Fall back to single-epoch metrics snapshot
        metrics = checkpoint.get("metrics", {})
        epoch = checkpoint.get("epoch", 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            f"Best epoch: {epoch}\n"
            + "\n".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items())),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
        )
        ax.set_title("Training Summary (no epoch history)")
        ax.axis("off")
    else:
        epochs = [h["epoch"] for h in epoch_history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Loss curves
        ax_loss = axes[0]
        if "loss_total" in epoch_history[0]:
            losses = [h["loss_total"] for h in epoch_history]
            ax_loss.plot(epochs, losses, label="train_loss")
        if "val_loss" in epoch_history[0]:
            val_losses = [h["val_loss"] for h in epoch_history]
            ax_loss.plot(epochs, val_losses, label="val_loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training & Validation Loss")
        ax_loss.legend()

        # Plot 2: R@10 curves
        ax_r10 = axes[1]
        r10_key = None
        for candidate in ["compound/mean_R@10", "mean_R@10"]:
            if candidate in epoch_history[0]:
                r10_key = candidate
                break
        if r10_key:
            r10_vals = [h[r10_key] for h in epoch_history]
            ax_r10.plot(epochs, r10_vals, label=r10_key)
        ax_r10.set_xlabel("Epoch")
        ax_r10.set_ylabel("R@10")
        ax_r10.set_title("Validation R@10")
        ax_r10.legend()

        fig.tight_layout()

    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved: %s", path)
    return path


def print_summary_table(metrics: dict[str, float]) -> None:
    """Print formatted summary table of all metrics to logger.

    Args:
        metrics: Full metrics dict from compute_all_metrics.
    """
    lines = ["=" * 70, "EVALUATION SUMMARY", "=" * 70]

    # Retrieval metrics — row level
    lines.append("\n--- Row-level Retrieval R@10 ---")
    for key in sorted(metrics.keys()):
        if (
            key.endswith("/R@10")
            and not key.startswith("compound/")
            and "->" in key
        ):
            direction = key.replace("/R@10", "")
            lines.append(f"  {direction:20s}  {metrics[key]:.4f}")
    if "mean_R@10" in metrics:
        lines.append(f"  {'mean':20s}  {metrics['mean_R@10']:.4f}")

    # Retrieval metrics — compound level
    lines.append("\n--- Compound-level Retrieval R@10 ---")
    for key in sorted(metrics.keys()):
        if (
            key.startswith("compound/")
            and key.endswith("/R@10")
            and "->" in key
        ):
            direction = key.split("/")[1]
            lines.append(f"  {direction:20s}  {metrics[key]:.4f}")
    if "compound/mean_R@10" in metrics:
        lines.append(
            f"  {'mean':20s}  {metrics['compound/mean_R@10']:.4f}"
        )

    # Diagnostics
    lines.append("\n--- Alignment & Uniformity ---")
    for key in sorted(metrics.keys()):
        if key.startswith("align_") or key.startswith("uniform_"):
            lines.append(f"  {key:20s}  {metrics[key]:.4f}")

    # MOA clustering
    moa_keys = [k for k in metrics if k.startswith("moa/")]
    if moa_keys:
        lines.append("\n--- MOA Clustering ---")
        for key in sorted(moa_keys):
            short = key.replace("moa/", "")
            lines.append(f"  {short:20s}  {metrics[key]:.4f}")

    lines.append("=" * 70)

    for line in lines:
        logger.info(line)


def generate_full_report(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str = "results/",
) -> Path:
    """Generate complete evaluation report (FR-8.4).

    Runs retrieval, alignment/uniformity, and MOA clustering.
    Generates:
    - results/retrieval_table.csv
    - results/retrieval_table.tex
    - results/umap_{modality}.png
    - results/similarity_heatmap.png
    - results/training_curves.png

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        test_data_path: Path to processed data directory (containing test.parquet
            and feature_columns.json).
        output_dir: Directory to write results to.

    Returns:
        Path to output directory.
    """
    from src.data.dataset import CaPyDataset, collate_fn

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, _config = load_model_and_config(checkpoint_path, device)

    # Build test data loader
    data_dir = Path(test_data_path)
    test_parquet = data_dir / "test.parquet"
    feature_columns = data_dir / "feature_columns.json"

    test_dataset = CaPyDataset(
        parquet_path=str(test_parquet),
        feature_columns_path=str(feature_columns),
        scarf_enabled=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Generate embeddings
    embeddings, compound_ids, moa_labels = generate_embeddings(
        model, test_loader, device
    )

    # Compute all metrics
    metrics = compute_all_metrics(embeddings, compound_ids, moa_labels)

    # Generate outputs
    generate_retrieval_table(metrics, output_path)
    generate_umap_plots(embeddings, moa_labels, output_path)
    generate_similarity_heatmap(embeddings, output_path)
    generate_training_curves(checkpoint_path, output_path)
    print_summary_table(metrics)

    logger.info("Full report generated in %s", output_path)
    return output_path
