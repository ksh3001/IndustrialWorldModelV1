"""Industrial World Model: seconds-ahead collision warnings from CCTV (hackathon POC)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml from the repo root (or a given path)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
