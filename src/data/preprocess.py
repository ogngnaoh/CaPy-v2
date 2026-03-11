"""Data preprocessing pipeline (FR-2.1 through FR-2.8).

Pipeline order:
1. Replicate correlation filtering (FR-2.1)
2. Replicate aggregation via MODZ (FR-2.2)
3. Treatment matching across modalities (FR-2.3)
4. Control removal (FR-2.4)
5. Feature QC (FR-2.5)
6. Scaffold splitting (FR-2.6)
7. Normalization (FR-2.7)
8. Save to parquet (FR-2.8)
"""

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def replicate_filter(
    df: pd.DataFrame, null_samples: int = 10000, percentile: int = 90
) -> pd.DataFrame:
    """Filter treatments by replicate correlation (FR-2.1)."""
    raise NotImplementedError


def aggregate_modz(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicates to treatment-level using MODZ (FR-2.2)."""
    raise NotImplementedError


def match_treatments(
    morph_df: pd.DataFrame, expr_df: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """Match treatments across modalities and merge SMILES (FR-2.3)."""
    raise NotImplementedError


def remove_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DMSO and vehicle controls (FR-2.4)."""
    raise NotImplementedError


def feature_qc(
    df: pd.DataFrame, nan_threshold: float = 0.05
) -> tuple[pd.DataFrame, dict]:
    """Run feature quality control (FR-2.5).

    Returns filtered dataframe and dict of retained feature column names.
    """
    raise NotImplementedError


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Split by Bemis-Murcko scaffold, grouped by compound (FR-2.6)."""
    raise NotImplementedError


def normalize(df: pd.DataFrame, feature_columns: dict) -> pd.DataFrame:
    """Normalize morphology (RobustScaler) and expression features (FR-2.7)."""
    raise NotImplementedError


def run_preprocessing_pipeline(config) -> None:
    """Execute full preprocessing pipeline (FR-2.1 through FR-2.8)."""
    raise NotImplementedError
