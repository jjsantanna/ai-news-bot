"""Rich-based structured logging shared across the package."""
from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "compliance_extractor") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("CE_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
