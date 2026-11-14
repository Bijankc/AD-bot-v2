"""When to keep a captured frame: on visible change (rate-limited), plus a slow periodic frame; within a disk budget.

Pure logic so it can be tested without a screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def thumb(img: np.ndarray) -> np.ndarray:
    """64x36 grey thumbnail; cheap to compare, ignores small noise."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)


def change(a: np.ndarray | None, b: np.ndarray) -> float:
    """Mean absolute difference of two thumbnails, 0-255. No previous thumbnail = maximal change."""
    if a is None or a.shape != b.shape:
        return 255.0
    return float(np.abs(a - b).mean())


@dataclass
class SavePolicy:
    change_threshold: float = 6.0  # mean grey difference that counts as a new picture
    min_gap: float = 2.0           # never save more often than this (seconds)
    max_gap: float = 20.0          # save at least this often while the game is in front
    max_bytes: int = 3 << 30       # stop after this much in one run
    min_free: int = 6 << 30        # stop when the disk has less free space than this

    def __post_init__(self):
        self._last_t: float | None = None
        self._last_thumb: np.ndarray | None = None
        self.bytes_written = 0

    def should_save(self, now: float, th: np.ndarray) -> tuple[bool, float]:
        d = change(self._last_thumb, th)
        if self._last_t is None:
            return True, d
        gap = now - self._last_t
        return (gap >= self.min_gap and d >= self.change_threshold) or gap >= self.max_gap, d

    def saved(self, now: float, th: np.ndarray, nbytes: int) -> None:
        self._last_t, self._last_thumb = now, th
        self.bytes_written += nbytes

    def out_of_budget(self, disk_free: int) -> str | None:
        if self.bytes_written >= self.max_bytes:
            return f"run budget reached ({self.bytes_written >> 20} MB)"
        if disk_free < self.min_free:
            return f"disk nearly full ({disk_free >> 30} GB free)"
        return None
