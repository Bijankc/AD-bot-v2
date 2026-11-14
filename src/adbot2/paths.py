"""Where large data lives. Kept outside the repo (and outside OneDrive) so it never syncs or gets committed."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """Root for venv-independent data: captures, datasets, models. Override with ADBOT2_HOME."""
    return Path(os.environ.get("ADBOT2_HOME", r"C:\adbot2"))


def data_dir() -> Path:
    return home() / "data"


def raw_dir() -> Path:
    """Imported frames, named by content hash."""
    return data_dir() / "raw"


def manifest_path() -> Path:
    return data_dir() / "manifest.jsonl"


def screen_labels_path() -> Path:
    return data_dir() / "labels" / "screens.jsonl"
