"""Tests for encoder and model architecture (FR-5)."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.capy import CaPyModel
from src.models.encoders import (
    ExpressionEncoder,
    MolecularEncoder,
    MorphologyEncoder,
)
from src.models.projections import ProjectionHead


@pytest.fixture
def tri_modal_config():
    """OmegaConf config for T1 (tri-modal)."""
    return OmegaConf.create(
        {
            "model": {
                "name": "tri_modal",
                "modalities": ["mol", "morph", "expr"],
                "mol_encoder": {
                    "input_dim": 2048,
                    "hidden_dims": [1024, 1024, 1024],
                    "output_dim": 512,
                    "dropout": 0.1,
                },
                "morph_encoder": {
                    "input_dim": 1500,
                    "hidden_dims": [1024, 1024],
                    "output_dim": 512,
                    "dropout": 0.1,
                },
                "expr_encoder": {
                    "input_dim": 978,
                    "hidden_dims": [1024, 1024],
                    "output_dim": 512,
                    "dropout": 0.1,
                },
                "projection": {
                    "input_dim": 512,
                    "hidden_dim": 512,
                    "output_dim": 256,
                },
                "embedding_dim": 256,
            }
        }
    )


@pytest.fixture
def bi_mol_morph_config():
    """OmegaConf config for B4 (mol + morph only)."""
    return OmegaConf.create(
        {
            "model": {
                "name": "bi_mol_morph",
                "modalities": ["mol", "morph"],
                "mol_encoder": {
                    "input_dim": 2048,
                    "hidden_dims": [1024, 1024, 1024],
                    "output_dim": 512,
                    "dropout": 0.1,
                },
                "morph_encoder": {
                    "input_dim": 1500,
                    "hidden_dims": [1024, 1024],
                    "output_dim": 512,
                    "dropout": 0.1,
                },
                "projection": {
                    "input_dim": 512,
                    "hidden_dim": 512,
                    "output_dim": 256,
                },
                "embedding_dim": 256,
            }
        }
    )


# ── Encoder tests (FR-5.1–5.3) ──────────────────────────────


class TestMolecularEncoder:
    """Tests for MolecularEncoder (FR-5.3)."""

    def test_output_shape(self):
        """Input [32, 2048] → output [32, 512]."""
        encoder = MolecularEncoder(input_dim=2048, output_dim=512, dropout=0.1)
        x = torch.randn(32, 2048)
        out = encoder(x)
        assert out.shape == (32, 512)

    def test_gradients_flow(self):
        """All parameters should have non-None gradients after backward."""
        encoder = MolecularEncoder(input_dim=2048, output_dim=512, dropout=0.1)
        x = torch.randn(32, 2048)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMorphologyEncoder:
    """Tests for MorphologyEncoder (FR-5.1)."""

    def test_output_shape(self):
        """Input [32, 1500] → output [32, 512]."""
        encoder = MorphologyEncoder(input_dim=1500, output_dim=512, dropout=0.1)
        x = torch.randn(32, 1500)
        out = encoder(x)
        assert out.shape == (32, 512)


class TestExpressionEncoder:
    """Tests for ExpressionEncoder (FR-5.2)."""

    def test_output_shape(self):
        """Input [32, 978] → output [32, 512]."""
        encoder = ExpressionEncoder(input_dim=978, output_dim=512, dropout=0.1)
        x = torch.randn(32, 978)
        out = encoder(x)
        assert out.shape == (32, 512)


# ── Projection head tests (FR-5.4) ──────────────────────────


class TestProjectionHead:
    """Tests for ProjectionHead (FR-5.4)."""

    def test_output_shape(self):
        """Input [32, 512] → output [32, 256]."""
        head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)
        x = torch.randn(32, 512)
        out = head(x)
        assert out.shape == (32, 256)

    def test_l2_normalized(self):
        """Output vectors should have unit L2 norm (±1e-6)."""
        head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)
        x = torch.randn(32, 512)
        out = head(x)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-6)


# ── CaPyModel tests (FR-5.5) ────────────────────────────────


class TestCaPyModel:
    """Tests for CaPyModel (FR-5.5)."""

    def test_tri_modal_has_all_encoders(self, tri_modal_config):
        """T1 config should have mol, morph, and expr encoders."""
        model = CaPyModel(tri_modal_config)
        assert "mol" in model.encoders
        assert "morph" in model.encoders
        assert "expr" in model.encoders
        assert "mol" in model.projections
        assert "morph" in model.projections
        assert "expr" in model.projections

    def test_bi_modal_missing_encoder(self, bi_mol_morph_config):
        """B4 config should NOT have expr encoder."""
        model = CaPyModel(bi_mol_morph_config)
        assert "mol" in model.encoders
        assert "morph" in model.encoders
        assert "expr" not in model.encoders
        assert "expr" not in model.projections

    def test_forward_output_shapes(self, tri_modal_config):
        """Forward pass returns correct embedding shapes."""
        model = CaPyModel(tri_modal_config)
        model.eval()
        batch = {
            "mol": torch.randn(16, 2048),
            "morph": torch.randn(16, 1500),
            "expr": torch.randn(16, 978),
        }
        embeddings, encoder_outputs = model(batch)
        assert embeddings["mol"].shape == (16, 256)
        assert embeddings["morph"].shape == (16, 256)
        assert embeddings["expr"].shape == (16, 256)
        assert encoder_outputs["mol"].shape == (16, 512)
        assert encoder_outputs["morph"].shape == (16, 512)
        assert encoder_outputs["expr"].shape == (16, 512)

    def test_forward_embeddings_l2_normalized(self, tri_modal_config):
        """All output embeddings should be L2-normalized."""
        model = CaPyModel(tri_modal_config)
        model.eval()
        batch = {
            "mol": torch.randn(16, 2048),
            "morph": torch.randn(16, 1500),
            "expr": torch.randn(16, 978),
        }
        embeddings, _ = model(batch)
        for m in ["mol", "morph", "expr"]:
            norms = torch.norm(embeddings[m], p=2, dim=1)
            assert torch.allclose(norms, torch.ones(16), atol=1e-6)

    def test_bi_modal_forward(self, bi_mol_morph_config):
        """B4 forward only returns mol and morph embeddings."""
        model = CaPyModel(bi_mol_morph_config)
        model.eval()
        batch = {
            "mol": torch.randn(8, 2048),
            "morph": torch.randn(8, 1500),
        }
        embeddings, encoder_outputs = model(batch)
        assert "mol" in embeddings
        assert "morph" in embeddings
        assert "expr" not in embeddings
        assert "mol" in encoder_outputs
        assert "morph" in encoder_outputs

    def test_parameter_count_reasonable(self, tri_modal_config):
        """Total parameter count should be in expected range."""
        model = CaPyModel(tri_modal_config)
        total = sum(p.numel() for p in model.parameters())
        # Rough: mol~4.7M + morph~3.1M + expr~2.6M + 3*proj~1.2M ≈ 11.6M
        assert total > 10_000_000
        assert total < 15_000_000

    def test_gradients_flow_end_to_end(self, tri_modal_config):
        """Gradients flow from loss through entire model."""
        model = CaPyModel(tri_modal_config)
        batch = {
            "mol": torch.randn(8, 2048),
            "morph": torch.randn(8, 1500),
            "expr": torch.randn(8, 978),
        }
        embeddings, encoder_outputs = model(batch)
        loss = sum(e.sum() for e in embeddings.values()) + sum(
            e.sum() for e in encoder_outputs.values()
        )
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
