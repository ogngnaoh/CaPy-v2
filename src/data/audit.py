"""Data audit module (FR-1.4).

Produces data/reports/lincs_audit.md with quality statistics
for all downloaded raw data sources.
"""

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


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
    """
    raise NotImplementedError
