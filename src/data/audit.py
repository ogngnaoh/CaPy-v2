"""Data audit module (FR-1.4).

Produces data/reports/lincs_audit.md with quality statistics
for all downloaded raw data sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _audit_morphology(morph_dir: Path) -> dict | None:
    """Audit morphology CSV files. Returns stats dict or None if missing."""
    morph_dir = Path(morph_dir)
    csv_files = sorted(
        list(morph_dir.rglob("*.csv")) + list(morph_dir.rglob("*.csv.gz"))
    )
    if not csv_files:
        logger.warning("No morphology CSV files found in %s", morph_dir)
        return None

    total_rows = 0
    total_nan = 0
    total_inf = 0
    total_cells = 0
    n_cols = 0
    col_names = set()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_file.name, exc)
            continue

        total_rows += len(df)
        numeric = df.select_dtypes(include=[np.number])
        n_cols = max(n_cols, numeric.shape[1])
        col_names.update(numeric.columns.tolist())
        total_nan += int(numeric.isna().sum().sum())
        total_inf += int(np.isinf(numeric.values).sum())
        total_cells += numeric.size

    nan_rate = total_nan / total_cells if total_cells > 0 else 0.0

    stats = {
        "n_files": len(csv_files),
        "total_rows": total_rows,
        "n_numeric_columns": n_cols,
        "nan_rate": nan_rate,
        "inf_count": total_inf,
    }
    logger.info(
        "Morphology: %d files, %d rows, %d numeric cols, NaN rate=%.4f, infs=%d",
        stats["n_files"],
        stats["total_rows"],
        stats["n_numeric_columns"],
        stats["nan_rate"],
        stats["inf_count"],
    )
    return stats


def _audit_expression(expr_dir: Path) -> dict | None:
    """Audit L1000 GCTX files. Returns stats dict or None if missing."""
    expr_dir = Path(expr_dir)
    if not expr_dir.exists():
        logger.warning("Expression directory not found: %s", expr_dir)
        return None

    stats = {}

    try:
        from cmapPy.pandasGEXpress.parse import parse
    except ImportError:
        logger.error(
            "cmapPy not installed — cannot audit GCTX files. "
            "Install with: pip install cmapPy"
        )
        return None

    for level, pattern in [("level_4", "level_4*.gctx"), ("level_5", "level_5*.gctx")]:
        gctx_files = list(expr_dir.glob(pattern))
        if not gctx_files:
            logger.warning("No %s GCTX file found in %s", level, expr_dir)
            continue

        gctx_path = gctx_files[0]
        try:
            gctoo = parse(str(gctx_path))
            df = gctoo.data_df
            stats[level] = {
                "file": gctx_path.name,
                "n_samples": df.shape[1],
                "n_genes": df.shape[0],
                "mean": float(df.values.mean()),
                "std": float(df.values.std()),
                "nan_rate": float(df.isna().sum().sum() / df.size),
            }
            logger.info(
                "%s: %d samples x %d genes, mean=%.3f, std=%.3f",
                level,
                stats[level]["n_samples"],
                stats[level]["n_genes"],
                stats[level]["mean"],
                stats[level]["std"],
            )
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", gctx_path.name, exc)

    # Audit column metadata files
    for meta_pattern in ["col_meta_*.txt", "REP.A_*.txt"]:
        for meta_file in expr_dir.glob(meta_pattern):
            try:
                meta_df = pd.read_csv(meta_file, sep="\t", low_memory=False)
                stats[meta_file.stem] = {
                    "file": meta_file.name,
                    "n_rows": len(meta_df),
                    "columns": meta_df.columns.tolist(),
                }
            except Exception as exc:
                logger.warning("Failed to read %s: %s", meta_file.name, exc)

    return stats if stats else None


def _audit_metadata(meta_path: Path) -> dict | None:
    """Audit Drug Repurposing Hub metadata. Returns stats dict or None."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        logger.warning("Metadata file not found: %s", meta_path)
        return None

    try:
        df = pd.read_csv(meta_path, sep="\t", comment="!", low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read metadata: %s", exc)
        return None

    n_compounds = len(df)
    smiles_col = "smiles" if "smiles" in df.columns else None
    broad_id_col = "broad_id" if "broad_id" in df.columns else None

    smiles_coverage = 0.0
    if smiles_col:
        smiles_coverage = 1.0 - df[smiles_col].isna().mean()

    moa_coverage = 0.0
    moa_col = "moa" if "moa" in df.columns else None
    if moa_col:
        moa_coverage = 1.0 - df[moa_col].isna().mean()

    stats = {
        "n_compounds": n_compounds,
        "columns": df.columns.tolist(),
        "smiles_coverage": smiles_coverage,
        "moa_coverage": moa_coverage,
        "has_broad_id": broad_id_col is not None,
        "has_smiles": smiles_col is not None,
    }
    logger.info(
        "Metadata: %d compounds, SMILES coverage=%.1f%%, MOA coverage=%.1f%%",
        n_compounds,
        smiles_coverage * 100,
        moa_coverage * 100,
    )
    return stats


def _compute_overlap(morph_dir: Path, expr_dir: Path, meta_path: Path) -> dict | None:
    """Compute treatment overlap across modalities using broad_id (13 chars)."""
    morph_ids = set()
    expr_ids = set()
    meta_ids = set()

    # Morphology: extract broad_id from CSVs
    morph_dir = Path(morph_dir)
    csv_files = list(morph_dir.rglob("*.csv")) + list(morph_dir.rglob("*.csv.gz"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(
                csv_file, usecols=lambda c: "broad" in c.lower(), low_memory=False
            )
            if df.empty:
                # Try reading Metadata_broad_sample column
                df = pd.read_csv(csv_file, low_memory=False)
            for col in df.columns:
                if "broad" in col.lower():
                    ids = df[col].dropna().astype(str).str[:13]
                    morph_ids.update(ids)
                    break
        except Exception:
            continue

    # Expression: extract compound IDs from column metadata
    # L1000 data uses "pert_id" (13-char BRD format), not "broad_id"
    expr_dir = Path(expr_dir)
    for meta_file in expr_dir.glob("col_meta_*.txt"):
        try:
            df = pd.read_csv(meta_file, sep="\t", low_memory=False)
            for col in df.columns:
                if "broad" in col.lower() or col.lower() == "pert_id":
                    ids = df[col].dropna().astype(str).str[:13]
                    expr_ids.update(ids)
                    break
        except Exception:
            continue

    # Also try pert_info file
    for pert_file in expr_dir.glob("*pert_info*"):
        try:
            df = pd.read_csv(pert_file, sep="\t", low_memory=False)
            for col in df.columns:
                if "broad" in col.lower() or col.lower() == "pert_id":
                    ids = df[col].dropna().astype(str).str[:13]
                    expr_ids.update(ids)
                    break
        except Exception:
            continue

    # Metadata
    meta_path = Path(meta_path)
    if meta_path.exists():
        try:
            df = pd.read_csv(meta_path, sep="\t", comment="!", low_memory=False)
            if "broad_id" in df.columns:
                ids = df["broad_id"].dropna().astype(str).str[:13]
                meta_ids.update(ids)
        except Exception:
            pass

    if not any([morph_ids, expr_ids, meta_ids]):
        return None

    overlap = {
        "morph_treatments": len(morph_ids),
        "expr_treatments": len(expr_ids),
        "meta_compounds": len(meta_ids),
    }

    if morph_ids and expr_ids:
        overlap["morph_expr_overlap"] = len(morph_ids & expr_ids)
    if morph_ids and meta_ids:
        overlap["morph_meta_overlap"] = len(morph_ids & meta_ids)
    if expr_ids and meta_ids:
        overlap["expr_meta_overlap"] = len(expr_ids & meta_ids)
    if morph_ids and expr_ids and meta_ids:
        overlap["all_three_overlap"] = len(morph_ids & expr_ids & meta_ids)

    return overlap


def _generate_report(
    morph_stats: dict | None,
    expr_stats: dict | None,
    meta_stats: dict | None,
    overlap: dict | None,
    output_path: Path,
) -> Path:
    """Write markdown audit report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# LINCS Data Audit Report\n"]

    # Morphology section
    lines.append("## Morphology (Cell Painting)\n")
    if morph_stats:
        lines.append(f"- **Files:** {morph_stats['n_files']}")
        lines.append(f"- **Total rows:** {morph_stats['total_rows']:,}")
        lines.append(f"- **Numeric columns:** {morph_stats['n_numeric_columns']}")
        lines.append(f"- **NaN rate:** {morph_stats['nan_rate']:.4f}")
        lines.append(f"- **Inf count:** {morph_stats['inf_count']}")
    else:
        lines.append("*Not available — morphology data not downloaded.*")
    lines.append("")

    # Expression section
    lines.append("## Expression (L1000)\n")
    if expr_stats:
        for level in ["level_4", "level_5"]:
            if level in expr_stats:
                s = expr_stats[level]
                lines.append(f"### {level.replace('_', ' ').title()}")
                lines.append(f"- **File:** {s['file']}")
                lines.append(f"- **Samples:** {s['n_samples']:,}")
                lines.append(f"- **Genes:** {s['n_genes']}")
                lines.append(f"- **Mean:** {s['mean']:.3f}")
                lines.append(f"- **Std:** {s['std']:.3f}")
                lines.append(f"- **NaN rate:** {s['nan_rate']:.4f}")
                lines.append("")
    else:
        lines.append("*Not available — expression data not downloaded.*")
    lines.append("")

    # Metadata section
    lines.append("## Metadata (Drug Repurposing Hub)\n")
    if meta_stats:
        lines.append(f"- **Compounds:** {meta_stats['n_compounds']:,}")
        lines.append(f"- **SMILES coverage:** {meta_stats['smiles_coverage']:.1%}")
        lines.append(f"- **MOA coverage:** {meta_stats['moa_coverage']:.1%}")
        lines.append(f"- **Has broad_id:** {meta_stats['has_broad_id']}")
        lines.append(f"- **Columns:** {', '.join(meta_stats['columns'][:15])}")
    else:
        lines.append("*Not available — metadata not downloaded.*")
    lines.append("")

    # Overlap section
    lines.append("## Cross-Modal Overlap\n")
    if overlap:
        if "morph_treatments" in overlap:
            lines.append(
                f"- **Morphology treatments:** {overlap['morph_treatments']:,}"
            )
        if "expr_treatments" in overlap:
            lines.append(f"- **Expression treatments:** {overlap['expr_treatments']:,}")
        if "meta_compounds" in overlap:
            lines.append(f"- **Metadata compounds:** {overlap['meta_compounds']:,}")
        if "morph_expr_overlap" in overlap:
            lines.append(f"- **Morph-Expr overlap:** {overlap['morph_expr_overlap']:,}")
        if "morph_meta_overlap" in overlap:
            lines.append(f"- **Morph-Meta overlap:** {overlap['morph_meta_overlap']:,}")
        if "expr_meta_overlap" in overlap:
            lines.append(f"- **Expr-Meta overlap:** {overlap['expr_meta_overlap']:,}")
        if "all_three_overlap" in overlap:
            lines.append(f"- **All three overlap:** {overlap['all_three_overlap']:,}")
    else:
        lines.append("*Cannot compute — insufficient data sources available.*")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    logger.info("Audit report written to %s", output_path)
    return output_path


def run_audit(
    morph_dir: str = "data/raw/morphology/",
    expr_dir: str = "data/raw/expression/",
    meta_path: str = "data/raw/metadata/repurposing_samples.txt",
    output_path: str = "data/reports/lincs_audit.md",
) -> Path:
    """Run full data audit and generate markdown report (FR-1.4).

    Report contains:
    - Morphology: row/column count, NaN rate histogram, inf count
    - Expression: row/column count, mean/std statistics
    - Metadata: compound count, SMILES coverage, MOA coverage
    - Cross-modal overlap: treatments present in both modalities

    Robust to missing sources — audits whatever is available.
    """
    morph_dir = Path(morph_dir)
    expr_dir = Path(expr_dir)
    meta_path_p = Path(meta_path)
    output_path_p = Path(output_path)

    logger.info("Starting data audit...")

    morph_stats = _audit_morphology(morph_dir)
    expr_stats = _audit_expression(expr_dir)
    meta_stats = _audit_metadata(meta_path_p)

    overlap = _compute_overlap(morph_dir, expr_dir, meta_path_p)

    report_path = _generate_report(
        morph_stats, expr_stats, meta_stats, overlap, output_path_p
    )

    # Summary warning
    paired_count = 0
    if overlap and "all_three_overlap" in overlap:
        paired_count = overlap["all_three_overlap"]
    elif overlap and "morph_expr_overlap" in overlap:
        paired_count = overlap["morph_expr_overlap"]

    if paired_count < 5000:
        logger.warning(
            "Cross-modal overlap is %d treatments (< 5000). "
            "This may be insufficient for contrastive training.",
            paired_count,
        )

    available = sum(1 for s in [morph_stats, expr_stats, meta_stats] if s is not None)
    logger.info(
        "Audit complete — %d/3 sources available, report at %s", available, report_path
    )

    return report_path
