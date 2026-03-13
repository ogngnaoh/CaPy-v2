"""Tests for data download, audit, dataset, and preprocessing modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
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


# ── Dataset tests (FR-4.1) ──────────────────────────────────


class TestCaPyDataset:
    """Tests for CaPyDataset (FR-4.1)."""

    def test_dataset_length_matches_parquet(self, mock_dataset_parquet):
        """Dataset length should equal parquet row count minus failed SMILES."""
        from src.data.dataset import CaPyDataset

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        assert len(ds) == mock_dataset_parquet["n_valid_rows"]

    def test_getitem_returns_correct_keys(self, mock_dataset_parquet):
        """Each item should have mol, morph, expr, metadata keys."""
        from src.data.dataset import CaPyDataset

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        item = ds[0]
        assert set(item.keys()) == {"mol", "morph", "expr", "metadata"}
        assert isinstance(item["metadata"], dict)
        assert "compound_id" in item["metadata"]
        assert "smiles" in item["metadata"]
        assert "moa" in item["metadata"]

    def test_tensor_shapes(self, mock_dataset_parquet):
        """mol=[2048], morph=[morph_dim], expr=[expr_dim]."""
        from src.data.dataset import CaPyDataset

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        item = ds[0]
        assert item["mol"].shape == (2048,)
        assert item["morph"].shape == (mock_dataset_parquet["morph_dim"],)
        assert item["expr"].shape == (mock_dataset_parquet["expr_dim"],)
        assert item["mol"].dtype == torch.float32
        assert item["morph"].dtype == torch.float32
        assert item["expr"].dtype == torch.float32

    def test_scarf_augmentation_training_only(self, mock_dataset_parquet):
        """SCARF should modify features during training, not eval."""
        from src.data.dataset import CaPyDataset

        ds_scarf = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
            scarf_enabled=True,
            scarf_corruption_rate=0.5,
        )
        ds_clean = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
            scarf_enabled=False,
        )
        # SCARF produces stochastic results
        torch.manual_seed(42)
        scarf_morph_1 = ds_scarf[0]["morph"]
        torch.manual_seed(123)
        scarf_morph_2 = ds_scarf[0]["morph"]
        clean_morph = ds_clean[0]["morph"]
        assert not torch.equal(scarf_morph_1, scarf_morph_2)
        assert clean_morph.shape == scarf_morph_1.shape

    def test_failed_smiles_excluded_and_logged(self, mock_dataset_parquet, caplog):
        """Failed SMILES are excluded and logged."""
        import logging

        from src.data.dataset import CaPyDataset

        with caplog.at_level(logging.WARNING):
            ds = CaPyDataset(
                parquet_path=mock_dataset_parquet["parquet_path"],
                feature_columns_path=mock_dataset_parquet["feature_columns_path"],
            )
        assert len(ds) == mock_dataset_parquet["n_valid_rows"]
        assert any(
            "excluded" in msg.lower() or "failed" in msg.lower()
            for msg in caplog.messages
        )

    def test_nan_filled_to_zero(self, mock_dataset_parquet):
        """No NaN in any output tensor."""
        from src.data.dataset import CaPyDataset

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        for i in range(len(ds)):
            item = ds[i]
            assert not torch.isnan(item["morph"]).any()
            assert not torch.isnan(item["expr"]).any()
            assert not torch.isnan(item["mol"]).any()

    def test_scarf_does_not_corrupt_mol(self, mock_dataset_parquet):
        """SCARF only corrupts morph/expr, not mol."""
        from src.data.dataset import CaPyDataset

        ds_scarf = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
            scarf_enabled=True,
            scarf_corruption_rate=0.5,
        )
        ds_clean = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
            scarf_enabled=False,
        )
        assert torch.equal(ds_scarf[0]["mol"], ds_clean[0]["mol"])


class TestCollate:
    """Tests for collate_fn."""

    def test_stacks_tensors(self, mock_dataset_parquet):
        """collate_fn stacks mol/morph/expr into batch tensors."""
        from src.data.dataset import CaPyDataset, collate_fn

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        batch = collate_fn([ds[0], ds[1]])
        assert batch["mol"].shape == (2, 2048)
        assert batch["morph"].shape == (2, mock_dataset_parquet["morph_dim"])
        assert batch["expr"].shape == (2, mock_dataset_parquet["expr_dim"])

    def test_collects_metadata(self, mock_dataset_parquet):
        """Metadata collected as list of dicts."""
        from src.data.dataset import CaPyDataset, collate_fn

        ds = CaPyDataset(
            parquet_path=mock_dataset_parquet["parquet_path"],
            feature_columns_path=mock_dataset_parquet["feature_columns_path"],
        )
        batch = collate_fn([ds[0], ds[1]])
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == 2
        assert "compound_id" in batch["metadata"][0]


class TestBuildDataloaders:
    """Tests for build_dataloaders (FR-4.2)."""

    def test_returns_three_loaders(self, mock_split_parquets):
        from src.data.dataset import build_dataloaders

        result = build_dataloaders(mock_split_parquets["config"])
        assert len(result) == 3

    def test_train_loader_drops_last(self, mock_split_parquets):
        from src.data.dataset import build_dataloaders

        train_dl, _, _ = build_dataloaders(mock_split_parquets["config"])
        ds_len = mock_split_parquets["train_len"]
        batch_size = mock_split_parquets["batch_size"]
        expected = ds_len // batch_size  # floor = drop_last
        assert len(train_dl) == expected

    def test_val_loader_keeps_last(self, mock_split_parquets):
        import math

        from src.data.dataset import build_dataloaders

        _, val_dl, _ = build_dataloaders(mock_split_parquets["config"])
        expected = math.ceil(
            mock_split_parquets["val_len"] / mock_split_parquets["batch_size"]
        )
        assert len(val_dl) == expected

    def test_train_loader_shuffles(self, mock_split_parquets):
        from torch.utils.data import RandomSampler

        from src.data.dataset import build_dataloaders

        train_dl, _, _ = build_dataloaders(mock_split_parquets["config"])
        assert isinstance(train_dl.sampler, RandomSampler)

    def test_val_loader_no_shuffle(self, mock_split_parquets):
        from torch.utils.data import SequentialSampler

        from src.data.dataset import build_dataloaders

        _, val_dl, _ = build_dataloaders(mock_split_parquets["config"])
        assert isinstance(val_dl.sampler, SequentialSampler)

    def test_batch_tensor_shapes(self, mock_split_parquets):
        from src.data.dataset import build_dataloaders

        train_dl, _, _ = build_dataloaders(mock_split_parquets["config"])
        batch = next(iter(train_dl))
        bs = mock_split_parquets["batch_size"]
        assert batch["mol"].shape == (bs, 2048)
        assert batch["morph"].shape[0] == bs
        assert batch["expr"].shape[0] == bs
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == bs


# ── Preprocessing: Data loaders (FR-2.0) ─────────────────────


class TestLoadMorphology:
    """Tests for _load_morphology."""

    def test_feature_intersection(self, mock_morph_batches):
        """Only shared features survive across batches."""
        from src.data.preprocess import _load_morphology

        df = _load_morphology(mock_morph_batches)
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))
        ]
        # 8 shared features, no batch-only features
        assert len(feature_cols) == 8
        assert not any("Extra_B1" in c for c in feature_cols)
        assert not any("Extra_B2" in c for c in feature_cols)

    def test_brd_truncation(self, mock_morph_batches):
        """22-char BRD IDs truncated to 13-char compound_id column."""
        from src.data.preprocess import _load_morphology

        df = _load_morphology(mock_morph_batches)
        assert "compound_id" in df.columns
        assert all(len(cid) == 13 for cid in df["compound_id"])

    def test_batch_concatenation(self, mock_morph_batches):
        """Both batches concatenated into single DataFrame."""
        from src.data.preprocess import _load_morphology

        df = _load_morphology(mock_morph_batches)
        assert len(df) == 60  # 30 + 30


class TestLoadExpression:
    """Tests for _load_expression."""

    def test_transpose_shape(self, mock_expr_data):
        """Output has samples as rows, genes as columns."""
        from src.data.preprocess import _load_expression

        expr_df, _ = _load_expression(
            mock_expr_data["dir"], data_df=mock_expr_data["data_df"]
        )
        # 40 samples as rows, gene columns present
        assert len(expr_df) == 40
        gene_cols = [c for c in expr_df.columns if c.endswith("_at")]
        assert len(gene_cols) == 20

    def test_col_meta_merged(self, mock_expr_data):
        """pert_id, x_smiles, dose_value present in output."""
        from src.data.preprocess import _load_expression

        expr_df, _ = _load_expression(
            mock_expr_data["dir"], data_df=mock_expr_data["data_df"]
        )
        assert "pert_id" in expr_df.columns
        assert "x_smiles" in expr_df.columns
        assert "dose_value" in expr_df.columns

    def test_pert_info_loaded(self, mock_expr_data):
        """Returns separate pert_info DataFrame with moa column."""
        from src.data.preprocess import _load_expression

        _, pert_info = _load_expression(
            mock_expr_data["dir"], data_df=mock_expr_data["data_df"]
        )
        assert "moa" in pert_info.columns
        assert "pert_id" in pert_info.columns
        assert len(pert_info) == 35

    def test_col_meta_without_inst_id(self, mock_expr_data_no_inst_id):
        """col_meta with non-inst_id first column is auto-renamed and merged."""
        from src.data.preprocess import _load_expression

        expr_df, _ = _load_expression(
            mock_expr_data_no_inst_id["dir"],
            data_df=mock_expr_data_no_inst_id["data_df"],
        )
        assert "pert_id" in expr_df.columns
        assert "x_smiles" in expr_df.columns
        assert len(expr_df) == 40


# ── Preprocessing: Pass-through stubs (FR-2.1/2.2) ──────────


class TestReplicateFilter:
    """Tests for replicate_filter pass-through."""

    def test_passthrough_returns_unchanged(self, mock_matched_df):
        """Output DataFrame identical to input."""
        from src.data.preprocess import replicate_filter

        result = replicate_filter(mock_matched_df)
        pd.testing.assert_frame_equal(result, mock_matched_df)

    def test_logs_passthrough_message(self, mock_matched_df, caplog):
        """Logs that data is pre-aggregated."""
        import logging

        from src.data.preprocess import replicate_filter

        with caplog.at_level(logging.INFO):
            replicate_filter(mock_matched_df)
        assert any("pre-aggregated" in msg.lower() for msg in caplog.messages)


class TestAggregateMODZ:
    """Tests for aggregate_modz pass-through."""

    def test_passthrough_returns_unchanged(self, mock_matched_df):
        """Output DataFrame identical to input."""
        from src.data.preprocess import aggregate_modz

        result = aggregate_modz(mock_matched_df)
        pd.testing.assert_frame_equal(result, mock_matched_df)

    def test_logs_passthrough_message(self, mock_matched_df, caplog):
        """Logs that data is pre-aggregated."""
        import logging

        from src.data.preprocess import aggregate_modz

        with caplog.at_level(logging.INFO):
            aggregate_modz(mock_matched_df)
        assert any("pre-aggregated" in msg.lower() for msg in caplog.messages)


# ── Preprocessing: Treatment matching (FR-2.3) ──────────────


class TestMatchTreatments:
    """Tests for match_treatments."""

    def test_inner_join_on_compound_id(
        self, mock_morph_batches, mock_expr_data, mock_metadata_for_preprocess
    ):
        """Only compounds in both morph and expr survive."""
        from src.data.preprocess import _load_expression, _load_morphology

        morph_df = _load_morphology(mock_morph_batches)
        expr_df, pert_info = _load_expression(
            mock_expr_data["dir"], data_df=mock_expr_data["data_df"]
        )
        meta_df = pd.read_csv(mock_metadata_for_preprocess, sep="\t")
        meta_df["compound_id"] = meta_df["broad_id"].str[:13]

        from src.data.preprocess import match_treatments

        result = match_treatments(morph_df, expr_df, meta_df, pert_info_df=pert_info)
        # All compound_ids must be in both morph and expr
        morph_ids = set(morph_df["compound_id"].unique())
        expr_ids = set(expr_df["pert_id"].str[:13].unique())
        for cid in result["compound_id"].unique():
            assert cid in morph_ids
            assert cid in expr_ids

    def test_cxsmiles_stripped(self):
        """CXSMILES annotation (after ' |') stripped before RDKit validation."""
        from src.data.preprocess import _strip_cxsmiles

        assert _strip_cxsmiles("CCO |SgD:0,1|") == "CCO"
        assert _strip_cxsmiles("CCO") == "CCO"

    def test_raises_on_missing_compound_and_pert_id(self):
        """Raises KeyError when expr_df has neither compound_id nor pert_id."""
        from src.data.preprocess import match_treatments

        morph_df = pd.DataFrame(
            {
                "compound_id": ["BRD-K00000001"],
                "Cells_F0": [1.0],
            }
        )
        # Expression df without compound_id or pert_id
        expr_df = pd.DataFrame(
            {
                "some_other_col": ["x"],
                "gene_0_at": [0.5],
            }
        )
        meta_df = pd.DataFrame(
            {
                "broad_id": ["BRD-K00000001"],
                "smiles": ["CCO"],
            }
        )
        with pytest.raises(KeyError, match="neither 'compound_id' nor 'pert_id'"):
            match_treatments(morph_df, expr_df, meta_df)

    def test_removes_missing_modality(
        self, mock_morph_batches, mock_expr_data, mock_metadata_for_preprocess
    ):
        """Treatments without valid SMILES dropped."""
        from src.data.preprocess import _load_expression, _load_morphology

        morph_df = _load_morphology(mock_morph_batches)
        expr_df, pert_info = _load_expression(
            mock_expr_data["dir"], data_df=mock_expr_data["data_df"]
        )
        meta_df = pd.read_csv(mock_metadata_for_preprocess, sep="\t")
        meta_df["compound_id"] = meta_df["broad_id"].str[:13]

        from src.data.preprocess import match_treatments

        result = match_treatments(morph_df, expr_df, meta_df, pert_info_df=pert_info)
        # No null SMILES
        assert result["smiles"].notna().all()


# ── Preprocessing: Control removal (FR-2.4) ─────────────────


class TestRemoveControls:
    """Tests for remove_controls."""

    def test_removes_dmso(self):
        """DMSO controls are removed."""
        from src.data.preprocess import remove_controls

        df = pd.DataFrame(
            {
                "pert_iname": ["DMSO", "cmpd_1", "cmpd_2"],
                "pert_type": ["ctl_vehicle", "trt_cp", "trt_cp"],
                "value": [1, 2, 3],
            }
        )
        result = remove_controls(df)
        assert "DMSO" not in result["pert_iname"].values

    def test_removes_empty(self):
        """Empty wells are removed."""
        from src.data.preprocess import remove_controls

        df = pd.DataFrame(
            {
                "pert_iname": ["EMPTY", "cmpd_1"],
                "pert_type": ["ctl_untrt", "trt_cp"],
                "value": [1, 2],
            }
        )
        result = remove_controls(df)
        assert len(result) == 1

    def test_case_insensitive(self):
        """Control removal is case-insensitive."""
        from src.data.preprocess import remove_controls

        df = pd.DataFrame(
            {
                "pert_iname": ["dmso", "Dmso", "cmpd_1"],
                "pert_type": ["trt_cp", "trt_cp", "trt_cp"],
                "value": [1, 2, 3],
            }
        )
        result = remove_controls(df)
        assert len(result) == 1

    def test_logs_removal_count(self, caplog):
        """Logs how many controls were removed."""
        import logging

        from src.data.preprocess import remove_controls

        df = pd.DataFrame(
            {
                "pert_iname": ["DMSO", "cmpd_1", "cmpd_2"],
                "pert_type": ["ctl_vehicle", "trt_cp", "trt_cp"],
                "value": [1, 2, 3],
            }
        )
        with caplog.at_level(logging.INFO):
            remove_controls(df)
        assert any("1" in msg and "control" in msg.lower() for msg in caplog.messages)


# ── Preprocessing: Feature QC (FR-2.5a) ─────────────────────


class TestFeatureQC:
    """Tests for feature_qc (phase A: pre-split)."""

    def test_inf_replaced_with_nan(self, mock_matched_df):
        """No inf values remain after QC."""
        import numpy as np

        from src.data.preprocess import feature_qc

        # Inject an inf
        mock_matched_df.loc[50, "Cells_Feature_3"] = np.inf
        result, _ = feature_qc(mock_matched_df)
        feature_cols = [
            c
            for c in result.columns
            if c.startswith("Cells_") or c.endswith("_at")
        ]
        assert not np.isinf(result[feature_cols].values).any()

    def test_nan_threshold_strictly_greater(self, mock_matched_df):
        """Feature under NaN threshold is kept."""
        from src.data.preprocess import feature_qc

        result, info = feature_qc(mock_matched_df, nan_threshold=0.05)
        # Cells_Feature_2 has ~4.1% NaN (4/97 after all-NaN row removal) → kept
        assert "Cells_Feature_2" in result.columns

    def test_high_nan_feature_removed(self, mock_matched_df):
        """Feature at >5% NaN is removed."""
        from src.data.preprocess import feature_qc

        result, info = feature_qc(mock_matched_df, nan_threshold=0.05)
        # Cells_Feature_1 has 6% NaN → removed
        assert "Cells_Feature_1" not in result.columns

    def test_all_nan_treatment_excluded(self, mock_matched_df):
        """All-NaN treatment rows dropped with warning."""
        from src.data.preprocess import feature_qc

        result, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        # Rows 97, 98, 99 were all-NaN → dropped
        assert len(result) < len(mock_matched_df)

    def test_detect_feature_columns(self):
        """Morph features by prefix, expr features by suffix."""
        from src.data.preprocess import _detect_feature_columns

        cols = [
            "compound_id",
            "Cells_Area",
            "Cytoplasm_Int",
            "Nuclei_Tex",
            "gene_0_at",
            "smiles",
        ]
        morph, expr = _detect_feature_columns(cols)
        assert morph == ["Cells_Area", "Cytoplasm_Int", "Nuclei_Tex"]
        assert expr == ["gene_0_at"]


# ── Preprocessing: Scaffold split (FR-2.6) ───────────────────


class TestScaffoldSplit:
    """Tests for scaffold_split."""

    def test_no_compound_leakage(self, mock_matched_df):
        """No compound_id appears in >1 split."""
        from src.data.preprocess import feature_qc, scaffold_split

        df, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        result = scaffold_split(df, seed=42)
        for cid in result["compound_id"].unique():
            splits = result[result["compound_id"] == cid]["split"].unique()
            assert len(splits) == 1, f"{cid} in multiple splits: {splits}"

    def test_proportions_within_tolerance(self, mock_matched_df):
        """Split proportions within +-12% of 70/15/15 (small dataset tolerance)."""
        from src.data.preprocess import feature_qc, scaffold_split

        df, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        result = scaffold_split(df, seed=42)
        n = len(result)
        n_train = (result["split"] == "train").sum()
        n_val = (result["split"] == "val").sum()
        n_test = (result["split"] == "test").sum()
        # Wider tolerance for small datasets — scaffold groups are indivisible
        assert abs(n_train / n - 0.70) < 0.12
        assert abs(n_val / n - 0.15) < 0.12
        assert abs(n_test / n - 0.15) < 0.12

    def test_all_doses_same_split(self):
        """Multiple rows of same compound in same split."""
        from src.data.preprocess import scaffold_split

        ring_smiles = [
            "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
            "c1cc2ccccc2cc1", "c1ccc2[nH]ccc2c1", "C1CCCCC1",
            "C1CCNCC1", "c1ccc(-c2ccccc2)cc1", "c1cnc2ccccc2n1",
            "c1ccc2ncccc2c1", "c1ccc2c(c1)ccc1ccccc12",
            "C1CCC2(CC1)CCCC2", "c1ccc(-c2ccccn2)cc1",
            "c1ccc2[nH]ncc2c1", "c1ccc2occc2c1", "c1ccc2sccc2c1",
            "C1CC2CCCC(C1)C2", "c1ccc(-c2ccco2)cc1",
            "c1ccc2c(c1)CCCC2", "c1ccnc(-c2ccccc2)c1",
            "c1ccc(-c2cccs2)cc1", "c1ccc2c(c1)occc2=O",
            "c1ccc(-c2ccc3ccccc3n2)cc1", "C1CCC(CC1)c1ccccc1",
            "c1ccc2c(c1)CC=C2", "c1ccc(-c2cnccn2)cc1",
        ]
        df = pd.DataFrame(
            {
                "compound_id": ["BRD-K00000001"] * 3
                + ["BRD-K00000002"] * 2
                + [f"BRD-K{i:08d}" for i in range(3, 30)],
                "smiles": ["c1ccccc1"] * 3
                + ["c1ccncc1"] * 2
                + ring_smiles[:27],
                "Cells_F0": np.random.randn(32),
            }
        )
        result = scaffold_split(df, seed=42)
        for cid in ["BRD-K00000001", "BRD-K00000002"]:
            splits = result[result["compound_id"] == cid]["split"].unique()
            assert len(splits) == 1

    def test_reproducible_with_seed(self, mock_matched_df):
        """Same seed -> same split."""
        from src.data.preprocess import feature_qc, scaffold_split

        df, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        r1 = scaffold_split(df.copy(), seed=42)
        r2 = scaffold_split(df.copy(), seed=42)
        assert (r1["split"].values == r2["split"].values).all()

    def test_seed_parameter_accepted(self, mock_matched_df):
        """scaffold_split accepts seed parameter without error."""
        from src.data.preprocess import feature_qc, scaffold_split

        df, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        result = scaffold_split(df, seed=123)
        assert "split" in result.columns
        assert set(result["split"].unique()) == {"train", "val", "test"}

    def test_adds_split_column(self, mock_matched_df):
        """Output has 'split' column with values train/val/test."""
        from src.data.preprocess import feature_qc, scaffold_split

        df, _ = feature_qc(mock_matched_df, nan_threshold=0.05)
        result = scaffold_split(df, seed=42)
        assert "split" in result.columns
        assert set(result["split"].unique()) == {"train", "val", "test"}


# ── Preprocessing: Zero-variance removal (FR-2.5b) ──────────


class TestZeroVarianceRemoval:
    """Tests for _remove_zero_variance."""

    def test_removes_zero_var_from_train(self, mock_matched_df):
        """Feature with std=0 on train set is removed from all splits."""
        from src.data.preprocess import _remove_zero_variance

        mock_matched_df["split"] = "train"
        morph_feats = [c for c in mock_matched_df.columns if c.startswith("Cells_")]
        expr_feats = [c for c in mock_matched_df.columns if c.endswith("_at")]
        result, new_morph, new_expr = _remove_zero_variance(
            mock_matched_df, morph_feats, expr_feats
        )
        # Cells_Feature_0 was constant (1.0) -> removed
        assert "Cells_Feature_0" not in result.columns
        assert "Cells_Feature_0" not in new_morph

    def test_nonzero_var_kept(self, mock_matched_df):
        """Features with nonzero variance on train set are kept."""
        from src.data.preprocess import _remove_zero_variance

        mock_matched_df["split"] = "train"
        morph_feats = [c for c in mock_matched_df.columns if c.startswith("Cells_")]
        expr_feats = [c for c in mock_matched_df.columns if c.endswith("_at")]
        result, new_morph, new_expr = _remove_zero_variance(
            mock_matched_df, morph_feats, expr_feats
        )
        # Cells_Feature_3 has random data -> kept
        assert "Cells_Feature_3" in result.columns
        assert "Cells_Feature_3" in new_morph
        # All expr features should be kept (all random)
        assert len(new_expr) == len(expr_feats)


# ── Preprocessing: Normalization (FR-2.7) ────────────────────


class TestNormalization:
    """Tests for normalize."""

    def _make_split_df(self):
        """Helper to create a split DataFrame for normalization tests."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "compound_id": [f"BRD-K{i:08d}" for i in range(n)],
                "smiles": [f"C{'C' * (i % 5)}O" for i in range(n)],
                "Cells_F0": np.random.randn(n) * 5 + 10,  # not centered
                "Cells_F1": np.random.randn(n) * 3 + 5,
                "gene_0_at": np.random.randn(n),  # already z-scored
                "gene_1_at": np.random.randn(n),
                "split": ["train"] * 70 + ["val"] * 15 + ["test"] * 15,
            }
        )
        return df

    def test_morph_robust_scaled(self):
        """Train morph has values roughly centered after scaling."""
        from src.data.preprocess import normalize

        df = self._make_split_df()
        feature_cols = {
            "morph_features": ["Cells_F0", "Cells_F1"],
            "expr_features": ["gene_0_at", "gene_1_at"],
        }
        result = normalize(df, feature_cols)
        train = result[result["split"] == "train"]
        # After RobustScaler, median should be near 0
        assert abs(train["Cells_F0"].median()) < 0.1

    def test_clipping_range(self):
        """All values within [-5, 5]."""
        from src.data.preprocess import normalize

        df = self._make_split_df()
        # Add extreme outlier
        df.loc[0, "Cells_F0"] = 1000.0
        feature_cols = {
            "morph_features": ["Cells_F0", "Cells_F1"],
            "expr_features": ["gene_0_at", "gene_1_at"],
        }
        result = normalize(df, feature_cols)
        all_feats = feature_cols["morph_features"] + feature_cols["expr_features"]
        for col in all_feats:
            assert result[col].max() <= 5.0
            assert result[col].min() >= -5.0

    def test_expr_already_zscored_not_rescaled(self):
        """Expression close to z-scored is only clipped."""
        from src.data.preprocess import normalize

        df = self._make_split_df()
        feature_cols = {
            "morph_features": ["Cells_F0", "Cells_F1"],
            "expr_features": ["gene_0_at", "gene_1_at"],
        }
        result = normalize(df, feature_cols)
        train = result[result["split"] == "train"]
        # Should still be close to z-scored
        assert abs(train["gene_0_at"].mean()) < 0.5
        assert abs(train["gene_0_at"].std() - 1.0) < 0.5

    def test_expr_drifted_is_rescaled(self):
        """Expression with mean=2.0 gets StandardScaler."""
        from src.data.preprocess import normalize

        df = self._make_split_df()
        df["gene_0_at"] = df["gene_0_at"] + 2.0  # drift the mean
        feature_cols = {
            "morph_features": ["Cells_F0", "Cells_F1"],
            "expr_features": ["gene_0_at", "gene_1_at"],
        }
        result = normalize(df, feature_cols)
        train = result[result["split"] == "train"]
        # After StandardScaler, mean should be near 0
        assert abs(train["gene_0_at"].mean()) < 0.1

    def test_scaler_fit_on_train_only(self):
        """Val/test transformed but not used for fitting."""
        from src.data.preprocess import normalize

        df = self._make_split_df()
        # Make val/test have very different distribution
        df.loc[df["split"] == "val", "Cells_F0"] = 100.0
        df.loc[df["split"] == "test", "Cells_F0"] = 100.0
        feature_cols = {
            "morph_features": ["Cells_F0", "Cells_F1"],
            "expr_features": ["gene_0_at", "gene_1_at"],
        }
        result = normalize(df, feature_cols)
        train = result[result["split"] == "train"]
        # Train stats should not be affected by val/test values
        assert abs(train["Cells_F0"].median()) < 0.5


# ── Preprocessing: Output + Pipeline (FR-2.8) ────────────────


class TestSaveOutputs:
    """Tests for _save_outputs."""

    def test_parquet_files_created(self, tmp_path):
        """Three parquet files created in output dir."""
        from src.data.preprocess import _save_outputs

        df = pd.DataFrame(
            {
                "compound_id": ["a", "b", "c"],
                "Cells_F0": [1.0, 2.0, 3.0],
                "gene_0_at": [0.1, 0.2, 0.3],
                "split": ["train", "val", "test"],
            }
        )
        feature_cols = {
            "morph_features": ["Cells_F0"],
            "expr_features": ["gene_0_at"],
        }
        _save_outputs(df, feature_cols, tmp_path)
        assert (tmp_path / "train.parquet").exists()
        assert (tmp_path / "val.parquet").exists()
        assert (tmp_path / "test.parquet").exists()

    def test_feature_columns_json(self, tmp_path):
        """JSON has morph_features and expr_features keys."""
        import json

        from src.data.preprocess import _save_outputs

        df = pd.DataFrame(
            {
                "compound_id": ["a", "b", "c"],
                "Cells_F0": [1.0, 2.0, 3.0],
                "gene_0_at": [0.1, 0.2, 0.3],
                "split": ["train", "val", "test"],
            }
        )
        feature_cols = {
            "morph_features": ["Cells_F0"],
            "expr_features": ["gene_0_at"],
        }
        _save_outputs(df, feature_cols, tmp_path)
        with open(tmp_path / "feature_columns.json") as f:
            data = json.load(f)
        assert "morph_features" in data
        assert "expr_features" in data
