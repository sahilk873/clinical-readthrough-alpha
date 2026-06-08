"""Structured logging configuration for clinical-alpha."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    name: str = "clinical_alpha",
) -> logging.Logger:
    """Configure structured logging with console and optional file output.

    Parameters
    ----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        If provided, write logs to this file.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logging()
