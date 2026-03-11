"""Tests for molecular featurization (FR-3.1)."""

import pytest


class TestSMILESToECFP:
    """Tests for smiles_to_ecfp (FR-3.1)."""

    def test_valid_smiles_shape(self):
        """SMILES 'CCO' (ethanol) should produce tensor of shape [2048]."""
        pytest.skip("Not yet implemented")

    def test_valid_smiles_dtype(self):
        """Output should be float32."""
        pytest.skip("Not yet implemented")

    def test_valid_smiles_nonzero_bits(self):
        """SMILES 'CCO' should have at least 3 nonzero bits."""
        pytest.skip("Not yet implemented")

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES 'INVALID' should return None."""
        pytest.skip("Not yet implemented")

    def test_cxsmiles_stripping(self):
        """SMILES with CXSMILES annotation should be parsed after stripping."""
        pytest.skip("Not yet implemented")
