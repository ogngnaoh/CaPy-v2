"""Shared test fixtures for CaPy v2."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_morph_csv(tmp_path):
    """Create a 20-row CSV with CellProfiler-style column names."""
    morph_dir = tmp_path / "morphology"
    morph_dir.mkdir()

    np.random.seed(42)
    n_rows = 20
    n_features = 10

    data = {
        "Metadata_broad_sample": [f"BRD-K{i:08d}" for i in range(n_rows)],
        "Metadata_Plate": ["plate1"] * n_rows,
        "Metadata_Well": [f"A{i:02d}" for i in range(n_rows)],
    }
    for j in range(n_features):
        data[f"Cells_AreaShape_Feature{j}"] = np.random.randn(n_rows)

    # Inject one NaN and one inf
    data["Cells_AreaShape_Feature0"][5] = np.nan
    data["Cells_AreaShape_Feature1"][10] = np.inf

    df = pd.DataFrame(data)
    csv_path = morph_dir / "profiles.csv"
    df.to_csv(csv_path, index=False)
    return morph_dir


@pytest.fixture
def mock_metadata_tsv(tmp_path):
    """Create a 15-row TSV with broad_id, smiles, moa columns."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()

    data = {
        "broad_id": [f"BRD-K{i:08d}" for i in range(15)],
        "pert_iname": [f"compound_{i}" for i in range(15)],
        "smiles": [f"C{'C' * i}O" for i in range(15)],
        "moa": [f"moa_{i % 5}" if i < 12 else None for i in range(15)],
        "target": [f"target_{i % 3}" for i in range(15)],
    }
    df = pd.DataFrame(data)
    meta_path = meta_dir / "repurposing_samples.txt"
    df.to_csv(meta_path, sep="\t", index=False)
    return meta_path


@pytest.fixture
def mock_download_config(tmp_path):
    """Create a config dict matching the updated lincs.yaml structure."""
    return {
        "sources": {
            "morphology": {
                "files": {
                    "batch1_modz": {
                        "url": "https://example.com/batch1_modz.csv.gz",
                        "filename": "batch1_consensus_modz.csv.gz",
                        "expected_size_mb": 66,
                    },
                    "batch2_modz": {
                        "url": "https://example.com/batch2_modz.csv.gz",
                        "filename": "batch2_consensus_modz.csv.gz",
                        "expected_size_mb": 78,
                    },
                },
                "local_path": str(tmp_path / "morphology"),
                "verification": {"min_rows": 15000, "min_columns": 500},
            },
            "expression": {
                "local_path": str(tmp_path / "expression"),
                "files": {
                    "level_5": {
                        "url": "https://example.com/level5.gctx",
                        "filename": "level_5_modz.gctx",
                        "primary": True,
                    },
                    "col_meta_level_5": {
                        "url": "https://example.com/col_meta.txt",
                        "filename": "col_meta_level_5.txt",
                        "primary": True,
                    },
                    "level_4": {
                        "url": "https://example.com/level4.gctx",
                        "filename": "level_4.gctx",
                        "primary": False,
                    },
                },
                "verification": {"min_rows": 5000, "min_columns": 978},
            },
            "metadata": {
                "primary_url": "https://example.com/meta.txt",
                "local_path": str(tmp_path / "metadata" / "repurposing_samples.txt"),
                "verification": {
                    "min_rows": 5000,
                    "required_columns": ["broad_id", "smiles"],
                },
            },
        },
        "download": {
            "retries": 1,
            "timeout_seconds": 5,
            "chunk_size_bytes": 1024,
        },
    }
