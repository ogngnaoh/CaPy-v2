"""Tests for training loop (FR-7)."""

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import collate_fn
from src.models.capy import CaPyModel
from src.models.losses import SigLIPLoss, VICRegLoss
from src.training.trainer import Trainer


# ── Fixtures ─────────────────────────────────────────────────


class _SyntheticDataset(Dataset):
    """Tiny synthetic dataset for trainer tests."""

    def __init__(
        self,
        n: int = 16,
        mol_dim: int = 32,
        morph_dim: int = 24,
        expr_dim: int = 16,
    ):
        torch.manual_seed(0)
        self.mol = torch.randn(n, mol_dim)
        self.morph = torch.randn(n, morph_dim)
        self.expr = torch.randn(n, expr_dim)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "mol": self.mol[idx],
            "morph": self.morph[idx],
            "expr": self.expr[idx],
            "metadata": {
                "compound_id": f"BRD-K{idx:08d}",
                "smiles": "CCO",
                "moa": "test",
            },
        }


def _make_tiny_config(tmp_path):
    return OmegaConf.create(
        {
            "seed": 42,
            "project_name": "test",
            "model": {
                "name": "test_model",
                "modalities": ["mol", "morph", "expr"],
                "mol_encoder": {
                    "input_dim": 32,
                    "hidden_dims": [64],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "morph_encoder": {
                    "input_dim": 24,
                    "hidden_dims": [64],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "expr_encoder": {
                    "input_dim": 16,
                    "hidden_dims": [64],
                    "output_dim": 32,
                    "dropout": 0.0,
                },
                "projection": {
                    "input_dim": 32,
                    "hidden_dim": 32,
                    "output_dim": 16,
                },
                "embedding_dim": 16,
            },
            "training": {
                "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
                "scheduler": {"name": "cosine", "warmup_epochs": 1},
                "batch_size": 4,
                "epochs": 3,
                "patience": 2,
                "gradient_clip_max_norm": 1.0,
                "loss": {"vicreg_lambda": 0.1, "siglib_bias_init": 0.0},
                "mixed_precision": False,
                "num_workers": 0,
            },
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )


@pytest.fixture
def trainer(tmp_path):
    """Build a Trainer with tiny model and synthetic data."""
    config = _make_tiny_config(tmp_path)
    model = CaPyModel(config)
    siglip_fn = SigLIPLoss(bias_init=config.training.loss.siglib_bias_init)
    vicreg_fn = VICRegLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(siglip_fn.parameters()),
        lr=config.training.optimizer.lr,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=1
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[1]
    )

    train_loader = DataLoader(
        _SyntheticDataset(n=16),
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        _SyntheticDataset(n=8, mol_dim=32, morph_dim=24, expr_dim=16),
        batch_size=4,
        collate_fn=collate_fn,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        siglip_loss_fn=siglip_fn,
        vicreg_loss_fn=vicreg_fn,
    )


# ── Unit tests ───────────────────────────────────────────────


class TestEarlyStopping:
    """Tests for check_early_stopping (FR-7.3)."""

    def test_returns_false_initially(self, trainer):
        """First call should not trigger early stopping."""
        assert trainer.check_early_stopping(0.5) is False

    def test_improvement_resets_counter(self, trainer):
        """Improvement should reset patience counter."""
        trainer.check_early_stopping(0.1)
        trainer.check_early_stopping(0.05)  # worse
        trainer.check_early_stopping(0.05)  # worse
        assert trainer.patience_counter == 2
        trainer.check_early_stopping(0.2)  # better
        assert trainer.patience_counter == 0

    def test_triggers_at_patience(self, trainer):
        """Should trigger after patience non-improving epochs."""
        trainer.check_early_stopping(0.5)  # sets best
        trainer.check_early_stopping(0.3)  # worse, counter=1
        result = trainer.check_early_stopping(0.3)  # worse, counter=2 >= patience=2
        assert result is True


class TestCollapseDetection:
    """Tests for check_collapse (FR-7.4)."""

    def test_warns_above_threshold(self, trainer, caplog):
        """Uniformity > -0.5 should trigger warning."""
        trainer.check_collapse({"uniform_mol": -0.3})
        assert "COLLAPSE WARNING" in caplog.text

    def test_silent_below_threshold(self, trainer, caplog):
        """Uniformity < -0.5 should not warn."""
        trainer.check_collapse({"uniform_mol": -2.5, "uniform_morph": -3.0})
        assert "COLLAPSE WARNING" not in caplog.text


class TestCheckpointing:
    """Tests for save_checkpoint (FR-7.2)."""

    def test_creates_file(self, trainer):
        """Checkpoint file should exist after save."""
        trainer.save_checkpoint(1, {"mean_R@10": 0.5})
        assert trainer.checkpoint_path.exists()

    def test_checkpoint_loadable(self, trainer):
        """Saved checkpoint should be loadable with expected keys."""
        trainer.save_checkpoint(1, {"mean_R@10": 0.5})
        checkpoint = torch.load(trainer.checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 1


# ── Integration tests ────────────────────────────────────────


class TestTrainEpoch:
    """Tests for train_epoch (FR-7.1)."""

    def test_returns_loss_dict(self, trainer):
        """train_epoch should return dict with loss_total."""
        metrics = trainer.train_epoch(1)
        assert "loss_total" in metrics
        assert isinstance(metrics["loss_total"], float)

    def test_loss_is_finite(self, trainer):
        """Loss should be finite (no NaN/inf)."""
        metrics = trainer.train_epoch(1)
        for key, val in metrics.items():
            assert torch.isfinite(torch.tensor(val)), f"{key} is not finite: {val}"

    def test_grad_norm_tracked(self, trainer):
        """Gradient norm should be tracked."""
        metrics = trainer.train_epoch(1)
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] > 0


class TestValidate:
    """Tests for validate (FR-7.1)."""

    def test_returns_retrieval_metrics(self, trainer):
        """validate should return dict with mean_R@10."""
        metrics = trainer.validate(1)
        assert "mean_R@10" in metrics
        assert 0.0 <= metrics["mean_R@10"] <= 1.0

    def test_returns_uniformity(self, trainer):
        """validate should return per-modality uniformity."""
        metrics = trainer.validate(1)
        for m in ["mol", "morph", "expr"]:
            assert f"uniform_{m}" in metrics

    def test_returns_val_loss(self, trainer):
        """validate should return val_loss."""
        metrics = trainer.validate(1)
        assert "val_loss" in metrics


class TestFullTraining:
    """Tests for complete train loop."""

    def test_train_completes(self, trainer):
        """Full training should complete and return best metrics."""
        result = trainer.train()
        assert "best_mean_R@10" in result
        assert "best_epoch" in result
        assert result["best_epoch"] >= 1
