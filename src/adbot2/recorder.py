"""When to keep a captured frame: on visible change (rate-limited), plus a slow periodic frame; within a disk budget.

Pure logic so it can be tested without a screen.

Thinning: the game is rarely static (fights change most of the picture every 2 s), so plain "changed since the last
save" keeps nearly everything. On top of it:
- a frame that nearly repeats any of the last `recent` saved frames is skipped (menus toggling back and forth,
  a results screen with one animated corner);
- on a static screen the "save anyway" gap doubles after each such save, up to `max_static_gap`.
A full-quality PNG is due every `png_every` seconds, regardless of thinning (for reading tests: exact pixels).
"""

from __future__ import annotations

from collections import deque
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
    max_gap: float = 20.0          # save at least this often while the game is in front...
    max_static_gap: float = 160.0  # ...doubling on a static screen up to this
    recent: int = 20               # near-duplicate check against this many recent saves
    png_every: float = 300.0       # a lossless frame this often (0 = never)
    max_bytes: int = 15 << 30      # stop after this much in one run
    min_free: int = 15 << 30       # stop when the disk has less free space than this

    def __post_init__(self):
        self._last_t: float | None = None
        self._last_png: float | None = None
        self._recent: deque[np.ndarray] = deque(maxlen=max(1, self.recent))
        self._static_run = 0       # consecutive saves made only because the gap ran out
        self.bytes_written = 0

    def gap_now(self) -> float:
        """The current "save anyway" gap (grows on static screens)."""
        return min(self.max_static_gap, self.max_gap * 2 ** self._static_run)

    def should_save(self, now: float, th: np.ndarray) -> tuple[bool, float, str]:
        """(keep, change vs the last save, kind).

        kind: first | change | periodic when kept; dup (changed, but repeats a recent save) or skip when not.
        """
        last = self._recent[-1] if self._recent else None
        d = change(last, th)
        if self._last_t is None:
            return True, d, "first"
        gap = now - self._last_t
        dup = False
        if gap >= self.min_gap and d >= self.change_threshold:
            if min(change(r, th) for r in self._recent) >= self.change_threshold:
                return True, d, "change"
            dup = True
        if gap >= self.gap_now():
            return True, d, "periodic"
        return False, d, "dup" if dup else "skip"

    def png_due(self, now: float) -> bool:
        return bool(self.png_every) and (self._last_png is None or now - self._last_png >= self.png_every)

    def saved(self, now: float, th: np.ndarray, nbytes: int, kind: str = "change", png: bool = False) -> None:
        self._static_run = self._static_run + 1 if kind == "periodic" else 0
        self._last_t = now
        self._recent.append(th)
        if png:
            self._last_png = now
        self.bytes_written += nbytes

    def out_of_budget(self, disk_free: int) -> str | None:
        if self.bytes_written >= self.max_bytes:
            return f"run budget reached ({self.bytes_written >> 20} MB)"
        if disk_free < self.min_free:
            return f"disk nearly full ({disk_free >> 30} GB free)"
        return None
