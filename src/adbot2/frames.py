"""Frame sources. Live capture and saved files share one interface so every test can run without the game."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Frame:
    image: np.ndarray        # BGR, HxWx3, uint8
    source: str              # file path or "live"
    t: float | None = None   # seconds since the source started, if known


def load_image(path: Path) -> np.ndarray:
    # cv2.imread cannot open non-ASCII paths on Windows; decode from bytes instead.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not an image: {path}")
    return img


class FileSource:
    """Frames from a folder (sorted by name) or an explicit list of files."""

    def __init__(self, paths: Path | list[Path]):
        if isinstance(paths, Path):
            self.paths = sorted(p for p in paths.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        else:
            self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for p in self.paths:
            yield Frame(load_image(p), str(p))
