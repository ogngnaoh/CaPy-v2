"""Entry point for data download (make setup).

Usage:
    python scripts/download.py                    # Download all sources
    python scripts/download.py --source morphology # Download one source
"""

import argparse
import sys

from src.data.audit import run_audit
from src.data.download import (
    download_all,
    download_expression,
    download_metadata,
    download_morphology,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

DOWNLOADERS = {
    "morphology": download_morphology,
    "expression": download_expression,
    "metadata": download_metadata,
}


def main():
    parser = argparse.ArgumentParser(description="Download LINCS data")
    parser.add_argument(
        "--source",
        choices=["morphology", "expression", "metadata"],
        default=None,
        help="Download a specific source. If omitted, downloads all.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip data audit after download.",
    )
    args = parser.parse_args()

    if args.source:
        try:
            DOWNLOADERS[args.source]()
        except Exception as exc:
            logger.error("Download failed for %s: %s", args.source, exc)
            sys.exit(1)
    else:
        errors = download_all()
        if errors:
            logger.error(
                "Download completed with %d error(s):\n  %s",
                len(errors),
                "\n  ".join(errors),
            )
            sys.exit(1)

    if not args.skip_audit:
        run_audit()


if __name__ == "__main__":
    main()
