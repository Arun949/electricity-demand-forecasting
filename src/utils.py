"""Shared helpers: config loading, logging, and project path resolution."""
import logging
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(config_path: str = "config/config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root, regardless of cwd."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / relative_path
