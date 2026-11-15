"""Shared logging setup.

Purpose:
    Give every module the same log format through one function, so the code
    uses ``logger.info(...)`` instead of bare ``print(...)``.

Main functions:
    - ``get_logger(name)``

Called by:
    ``utils.setup_resources``, every ``src`` module, and ``main.py``.
"""

from __future__ import annotations

import logging

from config.settings import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the project format and level.

    Args:
        name: Logger name. Use ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger``.

    Why: see knowledge.md, section "0. Supporting Code".
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(LOG_LEVEL)
    return logger
