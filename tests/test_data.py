"""Tests for dataset and preprocessing modules."""

import pytest


class TestCaPyDataset:
    """Tests for CaPyDataset (FR-4.1)."""

    def test_dataset_length_matches_parquet(self):
        """Dataset length should equal parquet row count minus failed SMILES."""
        pytest.skip("Not yet implemented")

    def test_getitem_returns_correct_keys(self):
        """Each item should have mol, morph, expr, metadata keys."""
        pytest.skip("Not yet implemented")

    def test_tensor_shapes(self):
        """mol=[2048], morph=[morph_dim], expr=[expr_dim]."""
        pytest.skip("Not yet implemented")

    def test_scarf_augmentation_training_only(self):
        """SCARF should modify features during training, not eval."""
        pytest.skip("Not yet implemented")


class TestPreprocessing:
    """Tests for preprocessing pipeline (FR-2)."""

    def test_replicate_filter_reduces_count(self):
        """Replicate filter should remove low-quality treatments."""
        pytest.skip("Not yet implemented")

    def test_scaffold_split_no_leakage(self):
        """No compound should appear in more than one split."""
        pytest.skip("Not yet implemented")

    def test_normalization_clipping(self):
        """All values should be within [-5, 5] after normalization."""
        pytest.skip("Not yet implemented")
