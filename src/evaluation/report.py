"""Full evaluation report generation (FR-8.4)."""

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_full_report(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str = "results/",
) -> Path:
    """Generate complete evaluation report (FR-8.4).

    Runs retrieval, alignment/uniformity, and MOA clustering.
    Generates:
    - results/retrieval_table.csv
    - results/retrieval_table.tex
    - results/umap_{modality}.png
    - results/similarity_heatmap.png
    - results/training_curves.png
    """
    raise NotImplementedError
