"""Tests for logging utilities (FR-11.0)."""

import logging

from src.utils.logging import (
    CAPY_DATEFMT,
    CAPY_FORMAT,
    get_logger,
    setup_file_logging,
    setup_log_level,
)


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        lg = get_logger("test.get_logger")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "test.get_logger"

    def test_format_matches_spec(self):
        lg = get_logger("test.format_check")
        handler = lg.handlers[0]
        assert CAPY_FORMAT in handler.formatter._fmt
        assert handler.formatter.datefmt == CAPY_DATEFMT


class TestSetupLogLevel:
    """Tests for setup_log_level."""

    def test_verbose_sets_debug(self):
        lg = get_logger("src.test_verbose")
        setup_log_level(logging.DEBUG)
        assert lg.level == logging.DEBUG
        # Reset
        setup_log_level(logging.INFO)

    def test_quiet_sets_warning(self):
        lg = get_logger("src.test_quiet")
        setup_log_level(logging.WARNING)
        assert lg.level == logging.WARNING
        # Reset
        setup_log_level(logging.INFO)


class TestSetupFileLogging:
    """Tests for setup_file_logging."""

    def test_creates_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        setup_file_logging(log_path)
        lg = get_logger("test.file_logging")
        lg.info("test message")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        assert log_path.exists()
        content = log_path.read_text()
        assert "test message" in content

        # Clean up: remove the file handler we added
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path):
                root.removeHandler(h)
                h.close()
