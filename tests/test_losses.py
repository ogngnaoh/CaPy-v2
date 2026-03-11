"""Tests for loss functions (FR-6)."""

import pytest


class TestSigLIPLoss:
    """Tests for SigLIPLoss (FR-6.1)."""

    def test_identical_embeddings_low_loss(self):
        """For z_a = z_b (perfect alignment), loss should be near 0."""
        pytest.skip("Not yet implemented")

    def test_random_embeddings_baseline_loss(self):
        """For random orthogonal embeddings, loss should be near log(2)."""
        pytest.skip("Not yet implemented")

    def test_output_is_scalar(self):
        """Loss should return a scalar tensor."""
        pytest.skip("Not yet implemented")


class TestVICRegLoss:
    """Tests for VICRegLoss (FR-6.2)."""

    def test_collapsed_embeddings_high_variance_loss(self):
        """All-identical embeddings → variance loss > 0."""
        pytest.skip("Not yet implemented")

    def test_well_distributed_low_variance_loss(self):
        """Well-distributed embeddings → variance loss ≈ 0."""
        pytest.skip("Not yet implemented")


class TestTotalLoss:
    """Tests for compute_total_loss (FR-6.3)."""

    def test_tri_modal_loss_keys(self):
        """T1 config should produce loss_mol_morph, loss_mol_expr, loss_morph_expr, vicreg_* keys."""
        pytest.skip("Not yet implemented")

    def test_bi_modal_loss_keys(self):
        """B4 config should only produce loss_mol_morph and vicreg_mol, vicreg_morph."""
        pytest.skip("Not yet implemented")
