"""Data preprocessing pipeline (FR-2.1 through FR-2.8).

Pipeline order:
1. Replicate correlation filtering (FR-2.1) — pass-through
2. Replicate aggregation via MODZ (FR-2.2) — pass-through
3. Treatment matching across modalities (FR-2.3)
4. Control removal (FR-2.4)
5. Feature QC phase A: inf/NaN (FR-2.5)
6. Scaffold splitting (FR-2.6)
7. Feature QC phase B: zero-variance on train (FR-2.5)
8. Normalization (FR-2.7)
9. Save to parquet (FR-2.8)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Module constants ────────────────────────────────────────────

_CELLPROFILER_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_")
_METADATA_PREFIX = "Metadata_"
_CONTROL_PATTERNS = {"dmso", "empty"}
_CONTROL_PERT_TYPES = {"ctl_vehicle", "ctl_untrt"}


# ── Feature detection ──────────────────────────────────────────


def _detect_feature_columns(
    columns: list[str],
) -> tuple[list[str], list[str]]:
    """Detect morph features (by prefix) and expr features (by suffix).

    Returns:
        (morph_features, expr_features)
    """
    morph = [c for c in columns if c.startswith(_CELLPROFILER_PREFIXES)]
    expr = [c for c in columns if c.endswith("_at")]
    return morph, expr


# ── Data loaders ────────────────────────────────────────────────


def _load_morphology(morph_dir: Path) -> pd.DataFrame:
    """Load and merge morphology CSVs, intersecting feature columns.

    Reads all CSVs from morph_dir. Computes intersection of feature columns
    across files. Truncates Metadata_broad_sample to 13-char compound_id.

    Returns:
        DataFrame with shared features + metadata columns.
    """
    morph_dir = Path(morph_dir)
    csv_files = sorted(morph_dir.glob("*.csv")) + sorted(morph_dir.glob("*.csv.gz"))
    if not csv_files:
        raise FileNotFoundError(f"No morphology CSVs found in {morph_dir}")

    dfs = []
    feature_sets = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        features = [c for c in df.columns if c.startswith(_CELLPROFILER_PREFIXES)]
        feature_sets.append(set(features))
        dfs.append(df)
        logger.info("Loaded %s: %d rows, %d features", f.name, len(df), len(features))

    # Intersect feature columns across all files
    shared_features = sorted(set.intersection(*feature_sets))
    logger.info(
        "Feature intersection: %d shared features across %d files",
        len(shared_features),
        len(csv_files),
    )

    # Identify metadata columns (present in all files)
    meta_sets = [
        {c for c in df.columns if c.startswith(_METADATA_PREFIX)} for df in dfs
    ]
    shared_meta = sorted(set.intersection(*meta_sets))

    keep_cols = shared_meta + shared_features
    merged = pd.concat([df[keep_cols] for df in dfs], ignore_index=True)

    # Truncate BRD IDs to 13 chars
    brd_col = "Metadata_broad_sample"
    if brd_col in merged.columns:
        merged["compound_id"] = merged[brd_col].astype(str).str[:13]
    else:
        raise KeyError(f"Expected column '{brd_col}' in morphology data")

    logger.info(
        "Morphology loaded: %d rows, %d shared features, %d unique compounds",
        len(merged),
        len(shared_features),
        merged["compound_id"].nunique(),
    )
    return merged


def _load_expression(
    expr_dir: Path,
    data_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load L1000 expression data, transpose, and merge column metadata.

    Args:
        expr_dir: Directory containing GCTX, col_meta, and pert_info files.
        data_df: Pre-loaded data DataFrame (genes×samples). If None, parse GCTX.

    Returns:
        (expr_df, pert_info_df) where expr_df has samples as rows with
        gene columns + metadata columns.
    """
    expr_dir = Path(expr_dir)
    gctoo = None

    # Load data matrix
    if data_df is None:
        try:
            from cmapPy.pandasGEXpress.parse import parse as parse_gctx
        except ImportError:
            raise ImportError(
                "cmapPy required for GCTX parsing. Install: pip install cmapPy"
            )
        gctx_files = list(expr_dir.glob("level_5*.gctx"))
        if not gctx_files:
            raise FileNotFoundError(f"No Level 5 GCTX file found in {expr_dir}")
        gctoo = parse_gctx(str(gctx_files[0]))
        data_df = gctoo.data_df

    # Transpose: genes-as-rows → samples-as-rows
    expr_t = data_df.T
    expr_t.index.name = "inst_id"
    expr_t = expr_t.reset_index()
    logger.info(
        "Expression matrix transposed: %d samples x %d genes",
        len(expr_t),
        len(expr_t.columns) - 1,
    )

    # Load and merge column metadata
    col_meta_files = list(expr_dir.glob("col_meta_*.txt"))
    col_meta = None
    if col_meta_files:
        col_meta = pd.read_csv(col_meta_files[0], sep="\t", low_memory=False)
        if "inst_id" not in col_meta.columns:
            # L1000 col_meta files often have sample IDs as unnamed first column
            first_col = col_meta.columns[0]
            sample_ids = set(expr_t["inst_id"].astype(str))
            if set(col_meta[first_col].astype(str).head(100)) & sample_ids:
                col_meta = col_meta.rename(columns={first_col: "inst_id"})
                logger.info("Renamed col_meta column '%s' to 'inst_id'", first_col)
            else:
                logger.warning(
                    "inst_id not found in col_meta columns: %s",
                    col_meta.columns.tolist()[:10],
                )
                col_meta = None

    # Fallback: GCTX embedded column metadata
    if col_meta is None and gctoo is not None:
        col_meta = gctoo.col_metadata_df.copy()
        col_meta.index.name = "inst_id"
        col_meta = col_meta.reset_index()
        logger.info(
            "Using GCTX embedded col_metadata (%d columns)",
            len(col_meta.columns) - 1,
        )

    if col_meta is not None:
        expr_t = expr_t.merge(col_meta, on="inst_id", how="left")
        logger.info("Merged col_meta: %d columns added", len(col_meta.columns) - 1)
    else:
        logger.warning("No column metadata available for expression data")

    # Load pert_info
    pert_info_files = list(expr_dir.glob("*pert_info*"))
    pert_info_df = pd.DataFrame()
    if pert_info_files:
        pert_info_df = pd.read_csv(pert_info_files[0], sep="\t", low_memory=False)
        logger.info("Loaded pert_info: %d entries", len(pert_info_df))

    return expr_t, pert_info_df


def _load_metadata(meta_path: Path) -> pd.DataFrame:
    """Load Drug Repurposing Hub metadata TSV.

    Truncates broad_id to 13 chars and adds compound_id column.
    """
    meta_path = Path(meta_path)
    df = pd.read_csv(meta_path, sep="\t", comment="!", low_memory=False)
    if "broad_id" in df.columns:
        df["compound_id"] = df["broad_id"].astype(str).str[:13]
    logger.info("Metadata loaded: %d rows", len(df))
    return df


# ── FR-2.1/2.2: Pass-through stubs ──────────────────────────────


def replicate_filter(
    df: pd.DataFrame, null_samples: int = 10000, percentile: int = 90
) -> pd.DataFrame:
    """Filter treatments by replicate correlation (FR-2.1).

    Pass-through: data arrives pre-aggregated as consensus MODZ profiles.
    """
    logger.info("FR-2.1 replicate_filter: pass-through (data is pre-aggregated MODZ)")
    return df


def aggregate_modz(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicates to treatment-level using MODZ (FR-2.2).

    Pass-through: data arrives pre-aggregated as consensus MODZ profiles.
    """
    logger.info("FR-2.2 aggregate_modz: pass-through (data is pre-aggregated MODZ)")
    return df


# ── FR-2.3: Treatment matching ──────────────────────────────────


def _strip_cxsmiles(smiles: str) -> str:
    """Strip CXSMILES annotation (everything after ' |')."""
    if not isinstance(smiles, str):
        return smiles
    idx = smiles.find(" |")
    if idx >= 0:
        return smiles[:idx]
    return smiles


def _validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string via RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    stripped = _strip_cxsmiles(smiles)
    # Skip known CMap sentinel values (e.g. -666) before calling RDKit
    # to avoid stderr spam from parse errors
    if stripped.lstrip("-").isdigit():
        return False
    mol = Chem.MolFromSmiles(stripped)
    return mol is not None


def _resolve_smiles(
    df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    pubchem_fallback: bool = False,
) -> pd.Series:
    """Resolve SMILES for each row via multi-stage fallback.

    Stage 1: Map compound_id → meta_df.compound_id → smiles
    Stage 2: Use x_smiles from expression col_meta (already in df)
    Stage 3 (optional): PubChem API lookup by compound name

    Returns:
        Series of resolved SMILES, NaN for failures.
    """
    smiles = pd.Series(index=df.index, dtype="object")
    resolved = pd.Series(False, index=df.index)

    # Stage 1: metadata lookup
    if meta_df is not None and "compound_id" in meta_df.columns:
        meta_smiles = meta_df.set_index("compound_id")["smiles"].to_dict()
        stage1 = df["compound_id"].map(meta_smiles)
        valid_mask = stage1.apply(_validate_smiles)
        smiles[valid_mask] = stage1[valid_mask].apply(_strip_cxsmiles)
        resolved[valid_mask] = True
        logger.info(
            "SMILES Stage 1 (metadata): %d / %d resolved",
            valid_mask.sum(),
            len(df),
        )

    # Stage 2: x_smiles from expression col_meta
    if "x_smiles" in df.columns:
        missing = ~resolved
        stage2 = df.loc[missing, "x_smiles"].apply(_strip_cxsmiles)
        valid_mask = stage2.apply(_validate_smiles)
        smiles.loc[stage2.index[valid_mask]] = stage2[valid_mask]
        resolved.loc[stage2.index[valid_mask]] = True
        logger.info(
            "SMILES Stage 2 (x_smiles): %d additional resolved",
            valid_mask.sum(),
        )

    # Stage 3: PubChem fallback (off by default)
    if pubchem_fallback:
        import time

        try:
            import pubchempy as pcp
        except ImportError:
            logger.warning("pubchempy not installed, skipping PubChem fallback")
            return smiles

        missing = ~resolved
        name_col = None
        for col in ["pert_iname", "Metadata_pert_iname"]:
            if col in df.columns:
                name_col = col
                break

        if name_col is not None:
            count = 0
            for idx in df.index[missing]:
                name = df.loc[idx, name_col]
                if not isinstance(name, str) or not name.strip():
                    continue
                try:
                    compounds = pcp.get_compounds(name, "name")
                    if compounds:
                        s = compounds[0].canonical_smiles
                        if _validate_smiles(s):
                            smiles[idx] = s
                            resolved[idx] = True
                            count += 1
                    time.sleep(0.5)
                except Exception:
                    continue
            logger.info("SMILES Stage 3 (PubChem): %d additional resolved", count)

    total_resolved = resolved.sum()
    total_missing = (~resolved).sum()
    logger.info(
        "SMILES resolution complete: %d resolved, %d missing (%.1f%% coverage)",
        total_resolved,
        total_missing,
        total_resolved / len(df) * 100 if len(df) > 0 else 0,
    )
    return smiles


def _resolve_moa(
    df: pd.DataFrame,
    pert_info_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Resolve MOA via union of pert_info and morphology metadata.

    Source 1: pert_info matched via compound_id (preferred)
    Source 2: Metadata_moa from morphology (gap filler)
    """
    moa = pd.Series(index=df.index, dtype="object")

    # Source 1: pert_info
    if pert_info_df is not None and "moa" in pert_info_df.columns:
        if "pert_id" in pert_info_df.columns:
            pert_info_df = pert_info_df.copy()
            pert_info_df["_cid"] = pert_info_df["pert_id"].astype(str).str[:13]
            moa_map = (
                pert_info_df.dropna(subset=["moa"]).set_index("_cid")["moa"].to_dict()
            )
            moa = df["compound_id"].map(moa_map)
            s1_count = moa.notna().sum()
            logger.info("MOA Source 1 (pert_info): %d resolved", s1_count)

    # Source 2: Metadata_moa from morphology
    if "Metadata_moa" in df.columns:
        missing = moa.isna()
        moa[missing] = df.loc[missing, "Metadata_moa"]
        s2_count = moa.notna().sum() - (~missing).sum()
        logger.info("MOA Source 2 (morph metadata): %d additional", max(0, s2_count))

    coverage = moa.notna().sum() / len(df) * 100 if len(df) > 0 else 0
    logger.info("MOA coverage: %.1f%%", coverage)
    return moa


def match_treatments(
    morph_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    pert_info_df: pd.DataFrame | None = None,
    pubchem_fallback: bool = False,
) -> pd.DataFrame:
    """Match treatments across modalities and merge SMILES (FR-2.3).

    Compound-level inner join on truncated 13-char BRD ID. Resolves SMILES
    via multi-stage fallback and MOA via union sourcing.

    Args:
        morph_df: Morphology DataFrame with compound_id column.
        expr_df: Expression DataFrame with pert_id column.
        meta_df: Metadata DataFrame with compound_id and smiles columns.
        pert_info_df: Optional pert_info for MOA resolution.
        pubchem_fallback: Whether to use PubChem API for SMILES fallback.

    Returns:
        Matched DataFrame with morph features, expr features, smiles, moa.
    """
    # Ensure compound_id on expression side
    if "compound_id" not in expr_df.columns:
        if "pert_id" in expr_df.columns:
            expr_df = expr_df.copy()
            expr_df["compound_id"] = expr_df["pert_id"].astype(str).str[:13]
        else:
            raise KeyError(
                "Expression DataFrame has neither 'compound_id' nor 'pert_id'. "
                f"Available columns: {expr_df.columns.tolist()[:15]}"
            )

    # Find overlapping compounds
    morph_ids = set(morph_df["compound_id"].unique())
    expr_ids = set(expr_df["compound_id"].unique())
    overlap = morph_ids & expr_ids
    logger.info(
        "Compound overlap: morph=%d, expr=%d, overlap=%d",
        len(morph_ids),
        len(expr_ids),
        len(overlap),
    )

    # Detect feature columns before merge
    morph_features = [
        c for c in morph_df.columns if c.startswith(_CELLPROFILER_PREFIXES)
    ]
    meta_cols = [c for c in morph_df.columns if c.startswith(_METADATA_PREFIX)]

    # Filter to overlapping compounds
    morph_overlap = morph_df[morph_df["compound_id"].isin(overlap)].copy()
    expr_filtered = expr_df[expr_df["compound_id"].isin(overlap)].copy()

    # Treatment-level pairing: assign morph profiles round-robin to expr rows
    # This preserves per-dose/batch morph variation instead of averaging
    keep_morph_cols = ["compound_id"] + morph_features + meta_cols
    morph_keep = morph_overlap[
        [c for c in keep_morph_cols if c in morph_overlap.columns]
    ].reset_index(drop=True)

    paired_rows = []
    for cid in sorted(overlap):
        expr_rows = expr_filtered[expr_filtered["compound_id"] == cid]
        morph_rows = morph_keep[morph_keep["compound_id"] == cid]
        n_expr = len(expr_rows)
        n_morph = len(morph_rows)
        if n_expr == 0 or n_morph == 0:
            continue
        # Round-robin: cycle morph profiles to match expr rows
        morph_cols_no_id = [c for c in morph_rows.columns if c != "compound_id"]
        morph_vals = morph_rows[morph_cols_no_id].values
        for i, (_, expr_row) in enumerate(expr_rows.iterrows()):
            morph_idx = i % n_morph
            row = expr_row.to_dict()
            for j, col in enumerate(morph_cols_no_id):
                row[col] = morph_vals[morph_idx, j]
            paired_rows.append(row)

    merged = pd.DataFrame(paired_rows)
    logger.info(
        "Treatment-level morph pairing: %d rows (%d unique morph profiles "
        "across %d compounds)",
        len(merged),
        len(morph_keep),
        len(overlap),
    )

    # Resolve SMILES
    if "compound_id" not in meta_df.columns and "broad_id" in meta_df.columns:
        meta_df = meta_df.copy()
        meta_df["compound_id"] = meta_df["broad_id"].astype(str).str[:13]

    merged["smiles"] = _resolve_smiles(
        merged, meta_df, pubchem_fallback=pubchem_fallback
    )

    # Resolve MOA
    merged["moa"] = _resolve_moa(merged, pert_info_df)

    # Drop rows without valid SMILES
    n_before = len(merged)
    merged = merged.dropna(subset=["smiles"]).reset_index(drop=True)
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        logger.info("Dropped %d rows with missing SMILES", n_dropped)

    logger.info(
        "Treatment matching complete: %d rows, %d unique compounds, "
        "SMILES coverage=%.1f%%, MOA coverage=%.1f%%",
        len(merged),
        merged["compound_id"].nunique(),
        merged["smiles"].notna().mean() * 100,
        merged["moa"].notna().mean() * 100,
    )
    return merged


# ── FR-2.4: Control removal ────────────────────────────────────


def remove_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DMSO and vehicle controls (FR-2.4).

    Checks pert_iname/Metadata_pert_iname for control patterns (case-insensitive)
    and pert_type for control types.
    """
    n_before = len(df)
    mask = pd.Series(False, index=df.index)

    # Check name columns for control patterns
    for col in ["pert_iname", "Metadata_pert_iname"]:
        if col in df.columns:
            name_lower = df[col].astype(str).str.lower().str.strip()
            for pattern in _CONTROL_PATTERNS:
                mask |= name_lower == pattern

    # Check pert_type for control types
    if "pert_type" in df.columns:
        type_lower = df["pert_type"].astype(str).str.lower().str.strip()
        for ctl_type in _CONTROL_PERT_TYPES:
            mask |= type_lower == ctl_type

    result = df[~mask].reset_index(drop=True)
    n_removed = n_before - len(result)
    logger.info(
        "Removed %d control treatment(s) (%d remaining)", n_removed, len(result)
    )
    return result


# ── FR-2.5: Feature QC ────────────────────────────────────────


def feature_qc(
    df: pd.DataFrame, nan_threshold: float = 0.05
) -> tuple[pd.DataFrame, dict]:
    """Run feature quality control phase A: pre-split (FR-2.5).

    Phase A (pre-split):
    - Replace inf/-inf with NaN
    - Drop all-NaN rows
    - Remove features with >nan_threshold NaN rate (strictly greater)

    Phase B (zero-variance) is handled separately by _remove_zero_variance.

    Returns:
        (filtered_df, {"morph_features": [...], "expr_features": [...]})
    """
    df = df.copy()
    morph_features, expr_features = _detect_feature_columns(df.columns.tolist())
    all_features = morph_features + expr_features

    # Replace inf/-inf with NaN
    n_inf = np.isinf(df[all_features].select_dtypes(include=[np.number])).sum().sum()
    if n_inf > 0:
        df[all_features] = df[all_features].replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced %d inf values with NaN", n_inf)

    # Drop all-NaN rows (where all feature columns are NaN)
    all_nan_mask = df[all_features].isna().all(axis=1)
    n_all_nan = all_nan_mask.sum()
    if n_all_nan > 0:
        logger.warning("Dropping %d all-NaN rows", n_all_nan)
        df = df[~all_nan_mask].reset_index(drop=True)

    # Remove features exceeding NaN threshold (strictly greater)
    nan_rates = df[all_features].isna().mean()
    high_nan = nan_rates[nan_rates > nan_threshold].index.tolist()
    if high_nan:
        logger.info(
            "Removing %d features with >%.1f%% NaN: %s",
            len(high_nan),
            nan_threshold * 100,
            high_nan[:5],
        )
        df = df.drop(columns=high_nan)

    # Recompute surviving features
    morph_features = [c for c in morph_features if c not in high_nan]
    expr_features = [c for c in expr_features if c not in high_nan]

    logger.info(
        "Feature QC phase A complete: %d morph + %d expr features, %d rows",
        len(morph_features),
        len(expr_features),
        len(df),
    )
    return df, {"morph_features": morph_features, "expr_features": expr_features}


def _remove_zero_variance(
    df: pd.DataFrame,
    morph_features: list[str],
    expr_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove zero-variance features based on training set (FR-2.5 phase B).

    Computes std on train split only. Drops columns with std == 0 from
    all splits. Returns updated df and surviving feature lists.
    """
    train_mask = df["split"] == "train"
    all_features = morph_features + expr_features

    train_std = df.loc[train_mask, all_features].std()
    zero_var = train_std[train_std == 0].index.tolist()

    if zero_var:
        logger.info(
            "Removing %d zero-variance features (on train): %s",
            len(zero_var),
            zero_var[:5],
        )
        df = df.drop(columns=zero_var)

    new_morph = [c for c in morph_features if c not in zero_var]
    new_expr = [c for c in expr_features if c not in zero_var]

    logger.info(
        "Feature QC phase B complete: %d morph + %d expr features retained",
        len(new_morph),
        len(new_expr),
    )
    return df, new_morph, new_expr


# ── FR-2.6: Scaffold splitting ─────────────────────────────────


def _get_scaffold(smiles: str) -> str:
    """Compute generic Bemis-Murcko scaffold.

    Falls back to original SMILES if scaffold extraction fails.
    """
    stripped = _strip_cxsmiles(smiles)
    try:
        mol = Chem.MolFromSmiles(stripped)
        if mol is None:
            return stripped
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic)
    except Exception:
        return stripped


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Split by Bemis-Murcko scaffold, grouped by compound (FR-2.6).

    All rows of the same compound go to the same split. Scaffolds are
    computed per unique compound, then groups are greedily assigned
    to the split furthest below target.

    Args:
        df: DataFrame with compound_id and smiles columns.
        train_frac, val_frac, test_frac: Target proportions.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 'split' column added.
    """
    df = df.copy()
    rng = random.Random(seed)

    # Get unique compounds with their SMILES
    compounds = df.drop_duplicates(subset=["compound_id"])[
        ["compound_id", "smiles"]
    ].copy()
    compounds["scaffold"] = compounds["smiles"].apply(_get_scaffold)

    # Group compounds by scaffold
    scaffold_groups = {}
    for _, row in compounds.iterrows():
        scaf = row["scaffold"]
        if scaf not in scaffold_groups:
            scaffold_groups[scaf] = []
        scaffold_groups[scaf].append(row["compound_id"])

    # Sort groups largest-first, break ties with seeded shuffle
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)  # shuffle first for tie-breaking
    groups.sort(key=len, reverse=True)

    # Count rows per compound for balanced splitting
    rows_per_compound = df.groupby("compound_id").size().to_dict()

    # Greedy assignment
    n_total = len(df)
    targets = {
        "train": train_frac * n_total,
        "val": val_frac * n_total,
        "test": test_frac * n_total,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    compound_to_split = {}

    for group in groups:
        group_size = sum(rows_per_compound.get(cid, 0) for cid in group)
        # Assign to split furthest below target
        best_split = min(counts, key=lambda s: counts[s] / max(targets[s], 1e-9))
        for cid in group:
            compound_to_split[cid] = best_split
        counts[best_split] += group_size

    df["split"] = df["compound_id"].map(compound_to_split)

    # Log split sizes
    for split_name in ["train", "val", "test"]:
        n = (df["split"] == split_name).sum()
        logger.info("Split %s: %d rows (%.1f%%)", split_name, n, n / len(df) * 100)

    return df


# ── FR-2.7: Normalization ──────────────────────────────────────


def normalize(df: pd.DataFrame, feature_columns: dict) -> pd.DataFrame:
    """Normalize morphology (RobustScaler) and expression features (FR-2.7).

    Morphology: RobustScaler fit on train, transform all, clip [-5, 5].
    Expression: Check if already z-scored. If not, StandardScaler. Clip [-5, 5].

    Args:
        df: DataFrame with 'split' column.
        feature_columns: Dict with 'morph_features' and 'expr_features' keys.

    Returns:
        Normalized DataFrame.
    """
    df = df.copy()
    train_mask = df["split"] == "train"
    morph_feats = feature_columns["morph_features"]
    expr_feats = feature_columns["expr_features"]

    # Morphology: RobustScaler
    if morph_feats:
        scaler = RobustScaler()
        # Fill NaN temporarily for scaling
        morph_data = df[morph_feats].copy()
        train_data = morph_data.loc[train_mask]
        scaler.fit(train_data.fillna(train_data.median()))
        morph_scaled = pd.DataFrame(
            scaler.transform(morph_data.fillna(train_data.median())),
            columns=morph_feats,
            index=df.index,
        )
        # Preserve NaN positions
        morph_scaled[morph_data.isna()] = np.nan
        df[morph_feats] = morph_scaled
        logger.info("Morphology: RobustScaler applied (%d features)", len(morph_feats))

    # Expression: conditional StandardScaler
    if expr_feats:
        train_expr = df.loc[train_mask, expr_feats]
        train_mean = train_expr.mean().mean()
        train_std = train_expr.std().mean()

        needs_rescale = abs(train_mean) > 0.1 or abs(train_std - 1.0) > 0.3
        if needs_rescale:
            scaler = StandardScaler()
            expr_data = df[expr_feats].copy()
            train_data = expr_data.loc[train_mask]
            scaler.fit(train_data.fillna(train_data.mean()))
            expr_scaled = pd.DataFrame(
                scaler.transform(expr_data.fillna(train_data.mean())),
                columns=expr_feats,
                index=df.index,
            )
            expr_scaled[expr_data.isna()] = np.nan
            df[expr_feats] = expr_scaled
            logger.info(
                "Expression: StandardScaler applied (mean=%.2f, std=%.2f)",
                train_mean,
                train_std,
            )
        else:
            logger.info(
                "Expression: already z-scored (mean=%.3f, std=%.3f), skipping rescale",
                train_mean,
                train_std,
            )

    # Clip all features to [-5, 5]
    all_feats = morph_feats + expr_feats
    df[all_feats] = df[all_feats].clip(-5.0, 5.0)
    logger.info("All features clipped to [-5, 5]")

    # Log per-split summary
    for split_name in ["train", "val", "test"]:
        split_data = df.loc[df["split"] == split_name, all_feats]
        logger.info(
            "  %s: mean=%.3f, std=%.3f",
            split_name,
            split_data.mean().mean(),
            split_data.std().mean(),
        )

    return df


# ── FR-2.8: Output ─────────────────────────────────────────────


def _save_outputs(
    df: pd.DataFrame,
    feature_columns: dict,
    output_dir: Path,
) -> None:
    """Save split DataFrames as parquet files and feature_columns.json.

    Args:
        df: Full DataFrame with 'split' column.
        feature_columns: Dict with morph_features and expr_features.
        output_dir: Directory to write output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        out_path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("Saved %s: %d rows -> %s", split_name, len(split_df), out_path)

    # Save feature column names
    json_path = output_dir / "feature_columns.json"
    with open(json_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    logger.info("Saved feature_columns.json -> %s", json_path)


# ── Pipeline orchestrator ──────────────────────────────────────


def run_preprocessing_pipeline(config) -> None:
    """Execute full preprocessing pipeline (FR-2.1 through FR-2.8).

    Args:
        config: OmegaConf DictConfig with sources, processing, split, output keys.
    """
    from omegaconf import OmegaConf

    if config is None:
        raise ValueError("Config required. Pass OmegaConf DictConfig.")

    logger.info("Starting preprocessing pipeline")

    # 1. Load data
    morph_dir = Path(config.sources.morphology.local_path)
    expr_dir = Path(config.sources.expression.local_path)
    meta_path = Path(config.sources.metadata.local_path)

    morph_df = _load_morphology(morph_dir)
    expr_df, pert_info_df = _load_expression(expr_dir)
    meta_df = _load_metadata(meta_path)

    # 2. Pass-through stubs (FR-2.1, FR-2.2)
    morph_df = replicate_filter(morph_df)
    morph_df = aggregate_modz(morph_df)
    expr_df = replicate_filter(expr_df)
    expr_df = aggregate_modz(expr_df)

    # 3. Treatment matching (FR-2.3)
    pubchem_fallback = OmegaConf.select(
        config, "processing.pubchem_fallback", default=False
    )
    matched = match_treatments(
        morph_df,
        expr_df,
        meta_df,
        pert_info_df=pert_info_df,
        pubchem_fallback=pubchem_fallback,
    )

    # 4. Control removal (FR-2.4)
    matched = remove_controls(matched)

    # 5. Feature QC phase A (FR-2.5)
    nan_threshold = OmegaConf.select(config, "processing.nan_threshold", default=0.05)
    matched, feature_cols = feature_qc(matched, nan_threshold=nan_threshold)

    # 6. Scaffold split (FR-2.6)
    seed = OmegaConf.select(config, "processing.seed", default=42)
    matched = scaffold_split(
        matched,
        train_frac=config.split.train,
        val_frac=config.split.val,
        test_frac=config.split.test,
        seed=seed,
    )

    # 7. Feature QC phase B: zero-variance (FR-2.5)
    matched, morph_feats, expr_feats = _remove_zero_variance(
        matched,
        feature_cols["morph_features"],
        feature_cols["expr_features"],
    )
    feature_cols = {"morph_features": morph_feats, "expr_features": expr_feats}

    # 8. Normalization (FR-2.7)
    matched = normalize(matched, feature_cols)

    # 9. Save outputs (FR-2.8)
    output_dir = Path(config.output.processed_dir)
    _save_outputs(matched, feature_cols, output_dir)

    logger.info(
        "Preprocessing pipeline complete: %d rows -> %s",
        len(matched),
        output_dir,
    )
