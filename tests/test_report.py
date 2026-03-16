"""Tests for full evaluation report (FR-8.4)."""

from pathlib import Path

import torch
import torch.nn.functional as f
from omegaconf import OmegaConf

from src.evaluation.report import (
    compute_all_metrics,
    generate_embeddings,
    generate_retrieval_table,
    generate_similarity_heatmap,
    generate_training_curves,
    generate_umap_plots,
    load_model_and_config,
    print_summary_table,
)


def _make_config():
    """Create a minimal config for testing."""
    return OmegaConf.create(
        {
            "model": {
                "name": "test_model",
                "modalities": ["mol", "morph", "expr"],
                "mol_encoder": {
                    "input_dim": 64,
                    "hidden_dims": [32],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "morph_encoder": {
                    "input_dim": 48,
                    "hidden_dims": [32],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "expr_encoder": {
                    "input_dim": 32,
                    "hidden_dims": [32],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "projection": {
                    "input_dim": 32,
                    "hidden_dim": 32,
                    "output_dim": 32,
                },
                "embedding_dim": 32,
            },
            "seed": 42,
        }
    )


def _make_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal checkpoint file for testing."""
    from src.models.capy import CaPyModel

    config = _make_config()
    model = CaPyModel(config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "siglip_state_dicts": {},
        "epoch": 10,
        "best_metric": 0.15,
        "metrics": {"mean_R@10": 0.15, "val_loss": 0.5},
        "config": OmegaConf.to_container(config, resolve=True),
        "epoch_history": [
            {
                "epoch": i,
                "loss_total": 1.0 - i * 0.05,
                "val_loss": 1.1 - i * 0.04,
                "mean_R@10": 0.05 + i * 0.01,
            }
            for i in range(1, 11)
        ],
    }
    path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, path)
    return path


class TestLoadModel:
    """Tests for load_model_and_config."""

    def test_load_model_from_checkpoint(self, tmp_path):
        """Should load model and config from saved checkpoint."""
        ckpt_path = _make_checkpoint(tmp_path)
        model, _config = load_model_and_config(str(ckpt_path), device="cpu")

        assert hasattr(model, "modalities")
        assert set(model.modalities) == {"mol", "morph", "expr"}
        assert not model.training  # Should be in eval mode


class TestRetrievalTable:
    """Tests for generate_retrieval_table."""

    def test_generate_retrieval_table_csv(self, tmp_path):
        """CSV file should exist and have expected columns."""
        metrics = {
            "mol->morph/R@1": 0.1,
            "mol->morph/R@5": 0.3,
            "mol->morph/R@10": 0.5,
            "mol->morph/MRR": 0.2,
            "morph->mol/R@1": 0.12,
            "morph->mol/R@5": 0.32,
            "morph->mol/R@10": 0.52,
            "morph->mol/MRR": 0.22,
            "compound/mol->morph/R@1": 0.15,
            "compound/mol->morph/R@5": 0.35,
            "compound/mol->morph/R@10": 0.55,
            "compound/mol->morph/MRR": 0.25,
        }
        csv_path, _tex_path = generate_retrieval_table(metrics, tmp_path)
        assert csv_path.exists()
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert "direction" in df.columns
        assert "row_R@10" in df.columns
        assert "compound_R@10" in df.columns

    def test_generate_retrieval_table_latex(self, tmp_path):
        """LaTeX file should exist and be non-empty."""
        metrics = {
            "mol->morph/R@10": 0.5,
            "mol->morph/MRR": 0.2,
        }
        _csv_path, tex_path = generate_retrieval_table(metrics, tmp_path)
        assert tex_path.exists()
        assert tex_path.stat().st_size > 0


class TestUMAPPlots:
    """Tests for generate_umap_plots."""

    def test_generate_umap_plots(self, tmp_path):
        """PNG files should be created for each modality."""
        torch.manual_seed(42)
        embeddings = {
            "mol": f.normalize(torch.randn(30, 32), dim=-1),
            "morph": f.normalize(torch.randn(30, 32), dim=-1),
            "expr": f.normalize(torch.randn(30, 32), dim=-1),
        }
        moa_labels = [f"moa_{i % 3}" if i < 20 else None for i in range(30)]
        paths = generate_umap_plots(embeddings, moa_labels, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_compute_all_metrics_keys(self):
        """Merged metrics dict should have retrieval + diagnostics + clustering keys."""
        torch.manual_seed(42)
        n = 40
        embeddings = {
            "mol": f.normalize(torch.randn(n, 32), dim=-1),
            "morph": f.normalize(torch.randn(n, 32), dim=-1),
            "expr": f.normalize(torch.randn(n, 32), dim=-1),
        }
        compound_ids = [f"BRD-{i // 2:04d}" for i in range(n)]
        moa_labels: list[str | None] = [f"moa_{i % 4}" for i in range(n)]

        metrics = compute_all_metrics(embeddings, compound_ids, moa_labels)

        # Retrieval keys
        assert "mean_R@10" in metrics
        assert "mol->morph/R@10" in metrics
        assert "compound/mean_R@10" in metrics

        # Diagnostics keys
        assert "align_mol_morph" in metrics
        assert "uniform_mol" in metrics

        # MOA keys
        assert "moa/AMI" in metrics
        assert "moa/ARI" in metrics


class TestPrintSummary:
    """Tests for print_summary_table."""

    def test_print_summary_table(self):
        """Should not crash and should complete without error."""
        metrics = {
            "mol->morph/R@10": 0.5,
            "morph->mol/R@10": 0.52,
            "mean_R@10": 0.51,
            "compound/mol->morph/R@10": 0.55,
            "compound/mean_R@10": 0.55,
            "align_mol_morph": 0.8,
            "uniform_mol": -3.0,
            "moa/AMI": 0.3,
            "moa/ARI": 0.25,
        }
        # Should not raise
        print_summary_table(metrics)


class TestGenerateEmbeddings:
    """Tests for generate_embeddings."""

    def test_returns_correct_shapes(self, tmp_path):
        """Embeddings should have [N, D] shape for each modality."""
        from torch.utils.data import DataLoader, Dataset

        from src.data.dataset import collate_fn
        from src.models.capy import CaPyModel

        config = _make_config()
        model = CaPyModel(config)
        model.eval()

        class _TinyDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "mol": torch.randn(64),
                    "morph": torch.randn(48),
                    "expr": torch.randn(32),
                    "metadata": {
                        "compound_id": f"BRD-{idx:04d}",
                        "smiles": "CCO",
                        "moa": "moa_0",
                    },
                }

        loader = DataLoader(_TinyDataset(), batch_size=4, collate_fn=collate_fn)
        embeddings, compound_ids, moa_labels = generate_embeddings(
            model, loader, device="cpu"
        )

        assert len(compound_ids) == 8
        assert len(moa_labels) == 8
        for m in ["mol", "morph", "expr"]:
            assert embeddings[m].shape == (8, 32)


class TestSimilarityHeatmap:
    """Tests for generate_similarity_heatmap."""

    def test_generates_png(self, tmp_path):
        """Should create similarity_heatmap.png."""
        torch.manual_seed(42)
        embeddings = {
            "mol": f.normalize(torch.randn(20, 32), dim=-1),
            "morph": f.normalize(torch.randn(20, 32), dim=-1),
            "expr": f.normalize(torch.randn(20, 32), dim=-1),
        }
        path = generate_similarity_heatmap(embeddings, tmp_path, max_samples=10)
        assert path.exists()
        assert path.suffix == ".png"


class TestTrainingCurves:
    """Tests for generate_training_curves."""

    def test_generate_training_curves_with_history(self, tmp_path):
        """Should generate PNG from epoch history."""
        ckpt_path = _make_checkpoint(tmp_path)
        path = generate_training_curves(str(ckpt_path), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"

    def test_generate_training_curves_no_history(self, tmp_path):
        """Should handle checkpoint without epoch_history gracefully."""
        checkpoint = {
            "epoch": 5,
            "metrics": {"mean_R@10": 0.12, "val_loss": 0.8},
        }
        ckpt_path = tmp_path / "no_history.pt"
        torch.save(checkpoint, ckpt_path)

        path = generate_training_curves(str(ckpt_path), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
