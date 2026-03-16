"""Logger setup (FR-11.0).

Format: {timestamp} | {module} | {level} | {message}
"""

import logging
import sys
from pathlib import Path

CAPY_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
CAPY_DATEFMT = "%Y-%m-%d %H:%M:%S"

_LOGGER_PREFIXES = ("src.", "scripts.", "__main__")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with the standard CaPy format.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(CAPY_FORMAT, datefmt=CAPY_DATEFMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def setup_log_level(level: int) -> None:
    """Set logging level on all CaPy loggers.

    Updates the root logger and any existing loggers whose names start
    with ``src.``, ``scripts.``, or ``__main__``.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.WARNING``).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and any(
            name.startswith(p) for p in _LOGGER_PREFIXES
        ):
            lg.setLevel(level)


def setup_file_logging(log_path: Path) -> None:
    """Add a file handler with the CaPy format to the root logger.

    Args:
        log_path: Path to the log file. Parent directories are created
            automatically.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(CAPY_FORMAT, datefmt=CAPY_DATEFMT)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
