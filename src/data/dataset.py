"""CaPy dataset and dataloader construction (FR-4.1, FR-4.2)."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.featurize import featurize_smiles_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CaPyDataset(Dataset):
    """Dataset returning aligned (mol, morph, expr) triplets (FR-4.1).

    Each __getitem__ returns a dict:
        {"mol": Tensor[2048], "morph": Tensor[morph_dim],
         "expr": Tensor[expr_dim], "metadata": {...}}

    SCARF augmentation is applied to morph and expr during training only.
    """

    def __init__(
        self,
        parquet_path: str,
        feature_columns_path: str,
        scarf_enabled: bool = False,
        scarf_corruption_rate: float = 0.4,
    ) -> None:
        # Load feature column lists
        with open(feature_columns_path) as f:
            feature_cols = json.load(f)
        morph_features = feature_cols["morph_features"]
        expr_features = feature_cols["expr_features"]

        # Load parquet
        df = pd.read_parquet(parquet_path)

        # Featurize SMILES
        smiles_list = df["smiles"].tolist()
        ecfp_dict = featurize_smiles_batch(smiles_list)

        # Filter to valid SMILES only
        valid_mask = df["smiles"].isin(ecfp_dict.keys())
        n_failed = (~valid_mask).sum()
        if n_failed > 0:
            logger.warning(
                "Excluded %d/%d treatments with failed SMILES featurization",
                n_failed,
                len(df),
            )
        df = df[valid_mask].reset_index(drop=True)

        # NaN check and fill
        n_nan_morph = df[morph_features].isna().sum().sum()
        n_nan_expr = df[expr_features].isna().sum().sum()
        if n_nan_morph + n_nan_expr > 0:
            logger.warning(
                "Filled %d NaN values with 0.0 (morph=%d, expr=%d)",
                n_nan_morph + n_nan_expr,
                n_nan_morph,
                n_nan_expr,
            )

        # Store tensors
        self._morph_data = torch.tensor(
            df[morph_features].fillna(0.0).values, dtype=torch.float32
        )
        self._expr_data = torch.tensor(
            df[expr_features].fillna(0.0).values, dtype=torch.float32
        )
        self._mol_data = torch.stack([ecfp_dict[smi] for smi in df["smiles"]])

        # Store metadata
        self._compound_ids = df["compound_id"].tolist()
        self._smiles_list = df["smiles"].tolist()
        self._moas = df["moa"].where(df["moa"].notna(), None).tolist()

        # SCARF config
        self._scarf_enabled = scarf_enabled
        self._scarf_corruption_rate = scarf_corruption_rate

        logger.info(
            "CaPyDataset: %d treatments, morph_dim=%d, expr_dim=%d, SCARF=%s",
            len(df),
            len(morph_features),
            len(expr_features),
            scarf_enabled,
        )

    def __len__(self) -> int:
        return len(self._morph_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        morph = self._morph_data[idx].clone()
        expr = self._expr_data[idx].clone()
        mol = self._mol_data[idx].clone()

        if self._scarf_enabled:
            mol = self._apply_ecfp_dropout(mol)
            morph = self._apply_scarf(morph, self._morph_data)
            expr = self._apply_scarf(expr, self._expr_data)

        return {
            "mol": mol,
            "morph": morph,
            "expr": expr,
            "metadata": {
                "compound_id": self._compound_ids[idx],
                "smiles": self._smiles_list[idx],
                "moa": self._moas[idx],
            },
        }

    def _apply_ecfp_dropout(self, mol: torch.Tensor) -> torch.Tensor:
        """Apply random bit dropout to ECFP fingerprints (10% of active bits)."""
        nonzero_idx = mol.nonzero(as_tuple=True)[0]
        n_nonzero = len(nonzero_idx)
        n_drop = int(n_nonzero * 0.1)
        if n_drop > 0:
            drop_sel = torch.randperm(n_nonzero)[:n_drop]
            mol[nonzero_idx[drop_sel]] = 0.0
        return mol

    def _apply_scarf(
        self, x: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        """Apply SCARF corruption by replacing features with empirical marginal draws."""
        d = x.shape[0]
        n_corrupt = int(d * self._scarf_corruption_rate)
        corrupt_indices = torch.randperm(d)[:n_corrupt]
        random_rows = torch.randint(0, data.shape[0], (n_corrupt,))
        x[corrupt_indices] = data[random_rows, corrupt_indices]
        return x


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that stacks tensors and collects metadata as list."""
    return {
        "mol": torch.stack([item["mol"] for item in batch]),
        "morph": torch.stack([item["morph"] for item in batch]),
        "expr": torch.stack([item["expr"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def build_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config (FR-4.2)."""
    processed_dir = Path(config.data.output.processed_dir)
    feature_columns_path = config.data.output.feature_columns_path

    scarf_enabled = config.training.scarf.enabled
    scarf_rate = config.training.scarf.corruption_rate
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    pin_memory = torch.cuda.is_available()

    train_ds = CaPyDataset(
        parquet_path=str(processed_dir / "train.parquet"),
        feature_columns_path=feature_columns_path,
        scarf_enabled=scarf_enabled,
        scarf_corruption_rate=scarf_rate,
    )
    val_ds = CaPyDataset(
        parquet_path=str(processed_dir / "val.parquet"),
        feature_columns_path=feature_columns_path,
        scarf_enabled=False,
    )
    test_ds = CaPyDataset(
        parquet_path=str(processed_dir / "test.parquet"),
        feature_columns_path=feature_columns_path,
        scarf_enabled=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    logger.info(
        "DataLoaders: train=%d batches (%d samples), "
        "val=%d batches (%d samples), test=%d batches (%d samples)",
        len(train_loader),
        len(train_ds),
        len(val_loader),
        len(val_ds),
        len(test_loader),
        len(test_ds),
    )
    return train_loader, val_loader, test_loader
