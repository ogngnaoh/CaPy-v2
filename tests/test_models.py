"""Tests for encoder and model architecture (FR-5)."""

import pytest


class TestMolecularEncoder:
    """Tests for MolecularEncoder (FR-5.3)."""

    def test_output_shape(self):
        """Input [32, 2048] → output [32, 512]."""
        pytest.skip("Not yet implemented")

    def test_gradients_flow(self):
        """All parameters should have non-None gradients after backward."""
        pytest.skip("Not yet implemented")


class TestMorphologyEncoder:
    """Tests for MorphologyEncoder (FR-5.1)."""

    def test_output_shape(self):
        """Input [32, 1500] → output [32, 512]."""
        pytest.skip("Not yet implemented")


class TestExpressionEncoder:
    """Tests for ExpressionEncoder (FR-5.2)."""

    def test_output_shape(self):
        """Input [32, 978] → output [32, 512]."""
        pytest.skip("Not yet implemented")


class TestProjectionHead:
    """Tests for ProjectionHead (FR-5.4)."""

    def test_output_shape(self):
        """Input [32, 512] → output [32, 256]."""
        pytest.skip("Not yet implemented")

    def test_l2_normalized(self):
        """Output vectors should have unit L2 norm (±1e-6)."""
        pytest.skip("Not yet implemented")


class TestCaPyModel:
    """Tests for CaPyModel (FR-5.5)."""

    def test_tri_modal_has_all_encoders(self):
        """T1 config should have mol, morph, and expr encoders."""
        pytest.skip("Not yet implemented")

    def test_bi_modal_missing_encoder(self):
        """B4 config should NOT have expr encoder."""
        pytest.skip("Not yet implemented")
