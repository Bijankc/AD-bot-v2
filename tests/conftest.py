import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def make_png(tmp_path):
    """Write a synthetic PNG (no real game data in tests). Returns its path."""
    def _make(name: str, size=(1920, 1080), value: int = 0, folder: Path | None = None) -> Path:
        d = folder or tmp_path / "src"
        d.mkdir(parents=True, exist_ok=True)
        img = np.full((size[1], size[0], 3), value, np.uint8)
        p = d / name
        cv2.imwrite(str(p), img)
        return p
    return _make
