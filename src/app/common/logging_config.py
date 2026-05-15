"""Structured logging configuration."""

import logging
import sys
import warnings

from app.config.settings import AppSettings, LogLevel

# Module-level logger — configured by setup_logging() at startup.
# Importable by all modules to avoid circular imports through app.main.
logger = logging.getLogger("email_extractor")


def setup_logging(settings: AppSettings) -> logging.Logger:
    """Configure structured logging for the application."""
    if not settings.debug:
        # Hide deprecation warnings in production.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger = logging.getLogger("email_extractor")
    logger.setLevel(logging.INFO if settings.log_level == LogLevel.INFO else logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
