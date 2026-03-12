"""Data download module for LINCS cpg0004 dataset (FR-1.1, FR-1.2, FR-1.3)."""

import sys
from pathlib import Path

import requests
from omegaconf import DictConfig, OmegaConf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data" / "lincs.yaml"


def _load_download_config() -> DictConfig:
    """Load configs/data/lincs.yaml as OmegaConf DictConfig."""
    cfg = OmegaConf.load(_CONFIG_PATH)
    assert isinstance(cfg, DictConfig)
    return cfg


def _build_session(retries: int = 3, timeout: int = 30) -> requests.Session:
    """Build a requests.Session with retry adapter."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_file(
    url: str,
    dest: Path,
    retries: int = 3,
    timeout: int = 30,
    chunk_size: int = 8192,
) -> Path:
    """Stream-download a file with atomic rename via .part file.

    Args:
        url: Source URL.
        dest: Final destination path.
        retries: Number of HTTP retries.
        timeout: Connection timeout in seconds.
        chunk_size: Download chunk size in bytes.

    Returns:
        Path to the downloaded file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    session = _build_session(retries=retries, timeout=timeout)
    logger.info("Downloading %s -> %s", url, dest)

    try:
        resp = session.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        sys.exit(1)

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(part, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                if pct % 25 == 0:
                    logger.info("  %d%% (%d / %d bytes)", pct, downloaded, total)

    part.rename(dest)
    logger.info("Saved %s (%d bytes)", dest.name, downloaded)
    return dest



def download_morphology(target_dir: str = "data/raw/morphology/") -> Path:
    """Download Cell Painting consensus MODZ profiles (FR-1.1).

    Downloads pre-aggregated treatment-level consensus profiles from the
    lincs-cell-painting GitHub repo (Git LFS). These are MODZ-aggregated
    full-feature profiles (~145 MB total for both batches).

    Skips files that already exist.
    """
    cfg = _load_download_config()
    morph_cfg = cfg.sources.morphology
    dl_cfg = cfg.download

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for _, file_info in morph_cfg.files.items():
        dest = target / file_info.filename
        if dest.exists():
            logger.info("Skipping %s — already exists", file_info.filename)
            continue

        _download_file(
            url=file_info.url,
            dest=dest,
            retries=dl_cfg.retries,
            timeout=dl_cfg.timeout_seconds,
            chunk_size=dl_cfg.chunk_size_bytes,
        )

    file_count = len(list(target.glob("*.csv.gz")))
    logger.info("Morphology download complete — %d files in %s", file_count, target)
    return target


def download_expression(
    target_dir: str = "data/raw/expression/",
    include_level4: bool = False,
) -> Path:
    """Download L1000 expression profiles from Figshare (FR-1.2).

    Downloads Level 5 (treatment-level MODZ) GCTX files plus column metadata.
    Level 4 (replicate-level) is optional and skipped by default.
    Skips files that already exist.

    Args:
        target_dir: Destination directory.
        include_level4: If True, also download Level 4 replicate-level files.
    """
    cfg = _load_download_config()
    expr_cfg = cfg.sources.expression
    dl_cfg = cfg.download

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for _, file_info in expr_cfg.files.items():
        is_primary = file_info.get("primary", True)
        if not is_primary and not include_level4:
            logger.info("Skipping %s (Level 4, not requested)", file_info.filename)
            continue

        dest = target / file_info.filename
        if dest.exists():
            logger.info("Skipping %s — already exists", file_info.filename)
            continue

        _download_file(
            url=file_info.url,
            dest=dest,
            retries=dl_cfg.retries,
            timeout=dl_cfg.timeout_seconds,
            chunk_size=dl_cfg.chunk_size_bytes,
        )

    logger.info("Expression download complete — files in %s", target)
    return target


def download_metadata() -> Path:
    """Download compound metadata from Drug Repurposing Hub (FR-1.3)."""
    cfg = _load_download_config()
    meta_cfg = cfg.sources.metadata
    dl_cfg = cfg.download

    dest = Path(meta_cfg.local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Metadata already exists at %s — skipping", dest)
        return dest

    _download_file(
        url=meta_cfg.primary_url,
        dest=dest,
        retries=dl_cfg.retries,
        timeout=dl_cfg.timeout_seconds,
        chunk_size=dl_cfg.chunk_size_bytes,
    )

    logger.info("Metadata download complete — %s", dest)
    return dest


def download_all() -> list[str]:
    """Download all three data sources.

    Returns list of error messages (empty if all succeeded).
    One failure does not block others.
    """
    errors = []

    for name, func in [
        ("morphology", download_morphology),
        ("expression", download_expression),
        ("metadata", download_metadata),
    ]:
        try:
            func()
        except SystemExit:
            errors.append(f"{name}: download failed (see logs above)")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.error("Failed to download %s: %s", name, exc)

    if errors:
        logger.error("Download errors:\n  %s", "\n  ".join(errors))
    else:
        logger.info("All downloads complete.")

    return errors
