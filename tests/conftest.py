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


# ── Preprocessing fixtures ────────────────────────────────────


@pytest.fixture
def mock_morph_batches(tmp_path):
    """Two CSVs simulating Batch 1 and Batch 2 with different column sets.

    Shared features (8): Cells_Area_F0..F3, Cytoplasm_Int_F0..F1, Nuclei_Tex_F0..F1
    Batch1-only (2): Cells_Extra_B1a, Cells_Extra_B1b
    Batch2-only (3): Cytoplasm_Extra_B2a, Cytoplasm_Extra_B2b, Nuclei_Extra_B2c
    30 rows each, 22-char BRD IDs, includes DMSO controls, NaN, inf.
    """
    morph_dir = tmp_path / "morphology"
    morph_dir.mkdir()
    np.random.seed(42)

    shared_features = [
        "Cells_Area_F0",
        "Cells_Area_F1",
        "Cells_Area_F2",
        "Cells_Area_F3",
        "Cytoplasm_Int_F0",
        "Cytoplasm_Int_F1",
        "Nuclei_Tex_F0",
        "Nuclei_Tex_F1",
    ]
    batch1_only = ["Cells_Extra_B1a", "Cells_Extra_B1b"]
    batch2_only = [
        "Cytoplasm_Extra_B2a",
        "Cytoplasm_Extra_B2b",
        "Nuclei_Extra_B2c",
    ]

    n = 30
    # 22-char BRD IDs (e.g. BRD-K76022557-003-28-9)
    brd_ids_1 = [f"BRD-K{i:08d}-001-01-1" for i in range(n)]
    brd_ids_2 = [f"BRD-K{i:08d}-002-02-2" for i in range(n)]
    # Make first 2 rows DMSO controls
    brd_ids_1[0] = "BRD-K00000000-001-01-1"
    brd_ids_2[0] = "BRD-K00000000-002-02-2"

    def _make_batch(brd_ids, extra_features, filename):
        data = {
            "Metadata_broad_sample": brd_ids,
            "Metadata_pert_iname": ["DMSO"] + [f"cmpd_{i}" for i in range(1, n)],
            "Metadata_moa": [None, None]
            + [f"moa_{i % 4}" for i in range(2, n)],
            "Metadata_Plate": [f"plate_{i % 3}" for i in range(n)],
        }
        for feat in shared_features + extra_features:
            data[feat] = np.random.randn(n)
        # Inject NaN and inf
        data[shared_features[0]][5] = np.nan
        data[shared_features[1]][10] = np.inf
        df = pd.DataFrame(data)
        df.to_csv(morph_dir / filename, index=False)

    _make_batch(brd_ids_1, batch1_only, "batch1_consensus_modz.csv")
    _make_batch(brd_ids_2, batch2_only, "batch2_consensus_modz.csv")

    return morph_dir


@pytest.fixture
def mock_expr_data(tmp_path):
    """Mock expression data: transposed DataFrame, col_meta TSV, pert_info TSV.

    data_df: 20 genes × 40 samples (genes-as-rows format).
    col_meta: 40 rows with pert_id, x_smiles, dose_value, pert_iname, pert_type.
    pert_info: 35 rows with pert_id, moa, pert_iname. ~80% MOA coverage.
    13-char BRD IDs, includes DMSO controls.
    """
    expr_dir = tmp_path / "expression"
    expr_dir.mkdir()
    np.random.seed(42)

    n_genes = 20
    n_samples = 40
    gene_ids = [f"gene_{i}_at" for i in range(n_genes)]
    # 13-char BRD IDs for samples
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    compound_ids = [f"BRD-K{i:08d}" for i in range(n_samples)]
    # First 2 are DMSO
    compound_ids[0] = "BRD-K00000000"
    compound_ids[1] = "BRD-K00000000"

    # data_df: genes as rows, samples as columns
    data_df = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=gene_ids,
        columns=sample_ids,
    )

    # col_meta
    smiles_list = [""] * 2 + [f"C{'C' * (i % 8)}O" for i in range(2, n_samples)]
    col_meta = pd.DataFrame(
        {
            "inst_id": sample_ids,
            "pert_id": compound_ids,
            "x_smiles": smiles_list,
            "dose_value": [0.0, 0.0] + [float(2**i) for i in range(n_samples - 2)],
            "pert_iname": ["DMSO", "DMSO"]
            + [f"cmpd_{i}" for i in range(2, n_samples)],
            "pert_type": ["ctl_vehicle", "ctl_vehicle"]
            + ["trt_cp"] * (n_samples - 2),
        }
    )
    col_meta.to_csv(
        expr_dir / "col_meta_level_5.txt", sep="\t", index=False
    )

    # pert_info — 35 unique compounds, ~80% with MOA
    n_pert = 35
    pert_info = pd.DataFrame(
        {
            "pert_id": [f"BRD-K{i:08d}" for i in range(n_pert)],
            "pert_iname": [f"cmpd_{i}" for i in range(n_pert)],
            "moa": [f"moa_{i % 5}" if i < 28 else None for i in range(n_pert)],
        }
    )
    pert_info.to_csv(
        expr_dir / "pert_info.txt", sep="\t", index=False
    )

    # Save data_df as a parquet for easy loading (mock for GCTX)
    data_df.to_parquet(expr_dir / "data_df.parquet")

    return {
        "dir": expr_dir,
        "data_df": data_df,
        "col_meta_path": expr_dir / "col_meta_level_5.txt",
        "pert_info_path": expr_dir / "pert_info.txt",
    }


@pytest.fixture
def mock_matched_df():
    """Pre-matched DataFrame with morph + expr features plus edge cases.

    100 rows, 15 morph features (Cells_*), 10 expr features (*_at),
    plus compound_id, smiles, moa, split (unset).
    Edge cases: one zero-variance morph feature, one >5% NaN feature,
    one under-threshold NaN feature, 3 all-NaN rows.
    Uses ring-containing SMILES for diverse Murcko scaffolds.
    """
    np.random.seed(42)
    n = 100
    n_morph = 15
    n_expr = 10

    # Ring-containing SMILES with diverse scaffolds for meaningful splitting
    scaffold_smiles = [
        "c1ccccc1",  # benzene
        "c1ccncc1",  # pyridine
        "c1ccoc1",  # furan
        "c1ccsc1",  # thiophene
        "c1cc2ccccc2cc1",  # naphthalene
        "c1ccc2[nH]ccc2c1",  # indole
        "C1CCCCC1",  # cyclohexane
        "C1CCNCC1",  # piperidine
        "c1ccc(-c2ccccc2)cc1",  # biphenyl
        "c1cnc2ccccc2n1",  # quinazoline
        "c1ccc2ncccc2c1",  # quinoline
        "c1ccc2c(c1)ccc1ccccc12",  # phenanthrene
        "C1CCC2(CC1)CCCC2",  # spiro[4.5]decane
        "c1ccc(-c2ccccn2)cc1",  # 2-phenylpyridine
        "c1ccc2[nH]ncc2c1",  # indazole
        "c1ccc2occc2c1",  # benzofuran
        "c1ccc2sccc2c1",  # benzothiophene
        "C1CC2CCCC(C1)C2",  # norbornane
        "c1ccc(-c2ccco2)cc1",  # 2-phenylfuran
        "c1ccc2c(c1)CCCC2",  # tetrahydronaphthalene
    ]

    data = {
        "compound_id": [f"BRD-K{i:08d}" for i in range(n)],
        "smiles": [scaffold_smiles[i % len(scaffold_smiles)] for i in range(n)],
        "moa": [f"moa_{i % 5}" if i < 80 else None for i in range(n)],
    }

    # Morph features
    for j in range(n_morph):
        col_name = f"Cells_Feature_{j}"
        data[col_name] = np.random.randn(n)

    # Expr features
    for j in range(n_expr):
        col_name = f"gene_{j}_at"
        data[col_name] = np.random.randn(n)

    df = pd.DataFrame(data)

    # Edge case 1: zero-variance morph feature (constant)
    df["Cells_Feature_0"] = 1.0

    # Edge case 2: >5% NaN feature (6 out of 100 = 6%)
    df.loc[0:5, "Cells_Feature_1"] = np.nan

    # Edge case 3: under-threshold NaN feature (4 out of 97 after dropping
    # all-NaN rows ≈ 4.1%, under 5% threshold → kept)
    df.loc[0:3, "Cells_Feature_2"] = np.nan

    # Edge case 4: 3 all-NaN rows (rows 97, 98, 99)
    morph_cols = [c for c in df.columns if c.startswith("Cells_")]
    expr_cols = [c for c in df.columns if c.endswith("_at")]
    for row_idx in [97, 98, 99]:
        df.loc[row_idx, morph_cols + expr_cols] = np.nan

    return df


@pytest.fixture
def mock_metadata_for_preprocess(tmp_path):
    """Metadata TSV for preprocessing tests with broad_id and smiles."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(exist_ok=True)

    data = {
        "broad_id": [f"BRD-K{i:08d}" for i in range(50)],
        "pert_iname": [f"compound_{i}" for i in range(50)],
        "smiles": [f"C{'C' * (i % 8)}O" for i in range(50)],
    }
    df = pd.DataFrame(data)
    meta_path = meta_dir / "repurposing_samples.txt"
    df.to_csv(meta_path, sep="\t", index=False)
    return meta_path


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


@pytest.fixture
def mock_dataset_parquet(tmp_path):
    """Processed parquet + feature_columns.json for CaPyDataset tests.

    20 rows, 5 morph features (Cells_*), 3 expr features (*_at).
    Row 19 has invalid SMILES ('INVALID_SMILES') to test exclusion.
    Row 18 has NaN in Cells_F0 to test NaN filling.
    """
    import json

    np.random.seed(42)
    n = 20
    morph_features = [f"Cells_F{i}" for i in range(5)]
    expr_features = [f"gene_{i}_at" for i in range(3)]

    valid_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCCC",
        "CC(C)O",
        "c1ccncc1",
        "CC=O",
        "CCCO",
        "c1ccoc1",
        "CC(=O)CC",
        "CCCCO",
        "c1ccsc1",
        "CC(O)CC",
        "CCCCC",
        "CCN",
        "c1ccc(O)cc1",
        "CCCCCO",
        "CC(N)C",
        "c1ccc(N)cc1",
    ]
    smiles_col = valid_smiles + ["INVALID_SMILES"]

    data = {
        "compound_id": [f"BRD-K{i:08d}" for i in range(n)],
        "smiles": smiles_col,
        "moa": [f"moa_{i % 3}" if i < 15 else None for i in range(n)],
    }
    for feat in morph_features:
        data[feat] = np.random.randn(n)
    for feat in expr_features:
        data[feat] = np.random.randn(n)

    df = pd.DataFrame(data)
    # NaN in one cell
    df.loc[18, "Cells_F0"] = np.nan

    parquet_path = tmp_path / "test_split.parquet"
    df.to_parquet(parquet_path, index=False)

    json_path = tmp_path / "feature_columns.json"
    with open(json_path, "w") as f:
        json.dump(
            {"morph_features": morph_features, "expr_features": expr_features}, f
        )

    return {
        "parquet_path": str(parquet_path),
        "feature_columns_path": str(json_path),
        "n_rows": n,
        "n_valid_rows": 19,  # 1 invalid SMILES excluded
        "morph_dim": 5,
        "expr_dim": 3,
    }


@pytest.fixture
def mock_split_parquets(tmp_path):
    """Train/val/test parquets + config for build_dataloaders tests."""
    import json

    from omegaconf import OmegaConf

    np.random.seed(42)
    morph_features = [f"Cells_F{i}" for i in range(5)]
    expr_features = [f"gene_{i}_at" for i in range(3)]

    valid_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCCC",
        "CC(C)O",
        "c1ccncc1",
        "CC=O",
        "CCCO",
        "c1ccoc1",
        "CC(=O)CC",
        "CCCCO",
        "c1ccsc1",
        "CC(O)CC",
        "CCCCC",
        "CCN",
        "c1ccc(O)cc1",
        "CCCCCO",
        "CC(N)C",
    ]

    splits = {"train": 16, "val": 6, "test": 6}
    idx = 0
    for split_name, n in splits.items():
        data = {
            "compound_id": [f"BRD-K{idx + i:08d}" for i in range(n)],
            "smiles": [
                valid_smiles[(idx + i) % len(valid_smiles)] for i in range(n)
            ],
            "moa": [f"moa_{i % 3}" for i in range(n)],
        }
        for feat in morph_features:
            data[feat] = np.random.randn(n)
        for feat in expr_features:
            data[feat] = np.random.randn(n)
        pd.DataFrame(data).to_parquet(
            tmp_path / f"{split_name}.parquet", index=False
        )
        idx += n

    json_path = tmp_path / "feature_columns.json"
    with open(json_path, "w") as f:
        json.dump(
            {"morph_features": morph_features, "expr_features": expr_features}, f
        )

    config = OmegaConf.create(
        {
            "data": {
                "output": {
                    "processed_dir": str(tmp_path),
                    "feature_columns_path": str(json_path),
                }
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "scarf": {"enabled": False, "corruption_rate": 0.4},
            },
        }
    )
    return {
        "config": config,
        "train_len": 16,
        "val_len": 6,
        "test_len": 6,
        "batch_size": 4,
    }
