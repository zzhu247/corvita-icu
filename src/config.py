"""Load config.yaml. Paths are resolved relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def p(rel: str) -> Path:
    """Project-relative path -> absolute path."""
    return ROOT / rel
