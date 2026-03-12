"""Tests for molecular featurization (FR-3.1)."""

import logging

import torch

from src.data.featurize import featurize_smiles_batch, smiles_to_ecfp


class TestSMILESToECFP:
    """Tests for smiles_to_ecfp (FR-3.1)."""

    def test_valid_smiles_shape(self):
        """SMILES 'CCO' (ethanol) should produce tensor of shape [2048]."""
        result = smiles_to_ecfp("CCO")
        assert result.shape == (2048,)

    def test_valid_smiles_dtype(self):
        """Output should be float32."""
        result = smiles_to_ecfp("CCO")
        assert result.dtype == torch.float32

    def test_valid_smiles_nonzero_bits(self):
        """SMILES 'CCO' should have at least 3 nonzero bits."""
        result = smiles_to_ecfp("CCO")
        assert result.sum().item() >= 3

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES 'INVALID' should return None."""
        result = smiles_to_ecfp("INVALID")
        assert result is None

    def test_cxsmiles_stripping(self):
        """SMILES with CXSMILES annotation should be parsed after stripping."""
        plain = smiles_to_ecfp("c1ccccc1")
        cxsmiles = smiles_to_ecfp("c1ccccc1 |SomeAnnotation|")
        assert plain is not None
        assert cxsmiles is not None
        assert torch.equal(plain, cxsmiles)

    def test_empty_string_returns_none(self):
        assert smiles_to_ecfp("") is None

    def test_none_input_returns_none(self):
        assert smiles_to_ecfp(None) is None

    def test_custom_n_bits(self):
        result = smiles_to_ecfp("CCO", n_bits=1024)
        assert result.shape == (1024,)

    def test_custom_radius(self):
        r2 = smiles_to_ecfp("c1ccc(CC)cc1", radius=2)
        r3 = smiles_to_ecfp("c1ccc(CC)cc1", radius=3)
        assert not torch.equal(r2, r3)

    def test_binary_values(self):
        result = smiles_to_ecfp("CCO")
        unique_vals = set(result.unique().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_warning_logged_for_invalid(self, caplog):
        with caplog.at_level(logging.WARNING):
            smiles_to_ecfp("INVALID")
        assert "Could not parse SMILES" in caplog.text


class TestFeaturizeSMILESBatch:
    """Tests for featurize_smiles_batch."""

    def test_returns_dict(self):
        """Returns dict mapping SMILES to tensor."""
        result = featurize_smiles_batch(["CCO", "c1ccccc1"])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_deduplication(self):
        """Duplicate SMILES are featurized only once."""
        result = featurize_smiles_batch(["CCO", "CCO", "c1ccccc1"])
        assert len(result) == 2

    def test_invalid_excluded(self):
        """Invalid SMILES not in result dict."""
        result = featurize_smiles_batch(["CCO", "INVALID"])
        assert "CCO" in result
        assert "INVALID" not in result

    def test_logs_summary(self, caplog):
        """Logs featurization summary per FSD spec."""
        with caplog.at_level(logging.INFO):
            featurize_smiles_batch(["CCO", "INVALID", "c1ccccc1"])
        assert "Featurized 2/3 compounds (1 failed)" in caplog.text
