"""Tests for data download, audit, and dataset modules."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.audit import (
    _audit_metadata,
    _audit_morphology,
    _generate_report,
    run_audit,
)
from src.data.download import (
    _load_download_config,
)

# ── Download helpers ────────────────────────────────────────────


class TestDownloadHelpers:
    """Tests for download helper functions."""

    def test_load_download_config(self):
        """Config loads and has expected top-level keys."""
        cfg = _load_download_config()
        assert "sources" in cfg
        assert "download" in cfg
        assert "morphology" in cfg.sources
        assert "expression" in cfg.sources
        assert "metadata" in cfg.sources

    def test_load_config_has_morphology_files(self):
        """Morphology config should have files dict with batch URLs."""
        cfg = _load_download_config()
        assert "files" in cfg.sources.morphology
        files = cfg.sources.morphology.files
        assert "batch1_modz" in files
        assert "batch2_modz" in files
        assert "url" in files.batch1_modz
        assert "filename" in files.batch1_modz

    def test_load_config_has_expression_files(self):
        """Expression config should list Level 5 as primary."""
        cfg = _load_download_config()
        files = cfg.sources.expression.files
        assert "level_5" in files
        assert files.level_5.get("primary", True) is True

    def test_load_config_level4_not_primary(self):
        """Level 4 should be marked as non-primary."""
        cfg = _load_download_config()
        files = cfg.sources.expression.files
        assert "level_4" in files
        assert files.level_4.get("primary", True) is False


# ── Download skip logic ────────────────────────────────────────


class TestDownloadSkipLogic:
    """Tests for skip-if-exists behavior."""

    @pytest.fixture(autouse=True)
    def _patch_config(self, mock_download_config, monkeypatch):
        """Patch config loader for all tests in this class."""
        monkeypatch.setattr(
            "src.data.download._load_download_config",
            lambda: OmegaConf.create(mock_download_config),
        )

    def test_morphology_skips_existing_files(self, tmp_path):
        """download_morphology skips files that already exist."""
        from src.data.download import download_morphology

        morph_dir = tmp_path / "morphology"
        morph_dir.mkdir()
        (morph_dir / "batch1_consensus_modz.csv.gz").write_bytes(b"fake")
        (morph_dir / "batch2_consensus_modz.csv.gz").write_bytes(b"fake")

        result = download_morphology(str(morph_dir))
        assert result == morph_dir

    def test_expression_skips_existing_files(self, tmp_path):
        """download_expression skips files that already exist."""
        from src.data.download import download_expression

        expr_dir = tmp_path / "expression"
        expr_dir.mkdir()
        (expr_dir / "level_5_modz.gctx").write_bytes(b"fake")
        (expr_dir / "col_meta_level_5.txt").write_bytes(b"fake")

        result = download_expression(str(expr_dir))
        assert result == expr_dir

    def test_expression_skips_level4_by_default(self, tmp_path, caplog):
        """download_expression skips Level 4 when include_level4=False."""
        import logging

        from src.data.download import download_expression

        expr_dir = tmp_path / "expression"
        expr_dir.mkdir()
        # Create all primary files so nothing is actually downloaded
        (expr_dir / "level_5_modz.gctx").write_bytes(b"fake")
        (expr_dir / "col_meta_level_5.txt").write_bytes(b"fake")

        with caplog.at_level(logging.INFO):
            download_expression(str(expr_dir), include_level4=False)

        assert any("Level 4" in msg for msg in caplog.messages)

    def test_metadata_skips_if_exists(self, tmp_path, mock_download_config):
        """download_metadata skips when file already exists."""
        from src.data.download import download_metadata

        meta_path = Path(mock_download_config["sources"]["metadata"]["local_path"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("broad_id\tsmiles\nBRD-K001\tCCO")

        result = download_metadata()
        assert result == meta_path


# ── Data audit ─────────────────────────────────────────────────


class TestDataAudit:
    """Tests for data audit functions."""

    def test_audit_morphology_stats(self, mock_morph_csv):
        """Morphology audit returns correct row count and NaN detection."""
        stats = _audit_morphology(mock_morph_csv)
        assert stats is not None
        assert stats["total_rows"] == 20
        assert stats["n_files"] == 1
        assert stats["nan_rate"] > 0  # We injected a NaN
        assert stats["inf_count"] >= 1  # We injected an inf

    def test_audit_morphology_missing(self, tmp_path):
        """Returns None when no morphology CSVs exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert _audit_morphology(empty_dir) is None

    def test_audit_metadata_columns(self, mock_metadata_tsv):
        """Metadata audit finds broad_id and smiles columns."""
        stats = _audit_metadata(mock_metadata_tsv)
        assert stats is not None
        assert stats["n_compounds"] == 15
        assert stats["has_broad_id"] is True
        assert stats["has_smiles"] is True
        assert stats["smiles_coverage"] == 1.0

    def test_audit_metadata_moa_coverage(self, mock_metadata_tsv):
        """MOA coverage reflects missing values."""
        stats = _audit_metadata(mock_metadata_tsv)
        assert stats is not None
        # 12/15 have MOA, 3 are None
        assert 0.7 < stats["moa_coverage"] < 0.9

    def test_audit_metadata_missing(self, tmp_path):
        """Returns None when metadata file doesn't exist."""
        assert _audit_metadata(tmp_path / "missing.txt") is None

    def test_generate_report(self, tmp_path):
        """Report is written with all 4 sections."""
        output = tmp_path / "report.md"
        morph_stats = {
            "n_files": 2,
            "total_rows": 1000,
            "n_numeric_columns": 500,
            "nan_rate": 0.01,
            "inf_count": 0,
        }
        meta_stats = {
            "n_compounds": 500,
            "columns": ["broad_id", "smiles", "moa"],
            "smiles_coverage": 0.95,
            "moa_coverage": 0.80,
            "has_broad_id": True,
            "has_smiles": True,
        }

        result = _generate_report(morph_stats, None, meta_stats, None, output)
        assert result.exists()
        content = result.read_text()
        assert "## Morphology" in content
        assert "## Expression" in content
        assert "## Metadata" in content
        assert "## Cross-Modal Overlap" in content

    def test_run_audit_warns_low_overlap(
        self, mock_morph_csv, mock_metadata_tsv, tmp_path, caplog
    ):
        """Audit warns when paired count < 5000."""
        import logging

        with caplog.at_level(logging.WARNING):
            run_audit(
                morph_dir=str(mock_morph_csv),
                expr_dir=str(tmp_path / "no_expr"),
                meta_path=str(mock_metadata_tsv),
                output_path=str(tmp_path / "report.md"),
            )
        assert any(
            "< 5000" in msg or "insufficient" in msg.lower() for msg in caplog.messages
        )

    def test_run_audit_creates_report(
        self, mock_morph_csv, mock_metadata_tsv, tmp_path
    ):
        """run_audit creates a report file even with partial data."""
        report = tmp_path / "audit.md"
        result = run_audit(
            morph_dir=str(mock_morph_csv),
            expr_dir=str(tmp_path / "no_expr"),
            meta_path=str(mock_metadata_tsv),
            output_path=str(report),
        )
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "Morphology" in content
        assert "Metadata" in content


# ── Dataset stubs (existing) ───────────────────────────────────


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
