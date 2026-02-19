"""
logger.py - Centralized logging configuration
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

try:
    import colorlog
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


def setup_logger(name: str = "trading_engine", log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if _HAS_COLOR:
        color_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(message)s",
            datefmt=datefmt,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(color_fmt)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    ch.setLevel(level)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    fh.setLevel(level)
    logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"trading_engine.{name}")
