"""Digit reader for the ENEMIES LEFT count: red digits on the HUD's opaque dark diamond, read at native pixels.

The diamond is opaque and the font is fixed (every digit 17 px tall, 7-16 px wide), so no OCR or network is needed:
red pixels -> connected glyphs (two touching digits are split) -> nearest mean shape per digit. A glyph far from
every shape, or a glyph count that makes no sense, gives no reading (None) rather than a guess: the old bot's OCR
errors ("41:28" read as "4:28") came from guessing.

Templates are built from the user's own frames (tools/digit_templates.py) and live under
<ADBOT2_HOME>/models/digits/; they are never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# number area inside the `enemies` crop (adbot2.crops.HEADS["enemies"], box 1800,795..1920,915)
NUM_Y0, NUM_Y1, NUM_X0, NUM_X1 = 48, 84, 36, 106
GLYPH_H, GLYPH_W = 17, 18      # template canvas; every digit is 17 px tall
MIN_H, MAX_ONE_W = 13, 17      # shorter blobs are noise; wider ones are touching digits


def red_mask(crop: np.ndarray) -> np.ndarray:
    """Red text pixels of a BGR crop's number area (uint8 0/1)."""
    b = crop[NUM_Y0:NUM_Y1, NUM_X0:NUM_X1].astype(np.int16)
    return ((b[..., 2] > 150) & (b[..., 2] - b[..., 1] > 70)).astype(np.uint8)


def _place(g: np.ndarray) -> np.ndarray:
    """A glyph mask on the fixed canvas: top-left aligned, cropped to the canvas."""
    out = np.zeros((GLYPH_H, GLYPH_W), np.float32)
    h, w = min(g.shape[0], GLYPH_H), min(g.shape[1], GLYPH_W)
    out[:h, :w] = g[:h, :w]
    return out


def glyphs(crop: np.ndarray) -> list[np.ndarray]:
    """Digit glyphs left to right, each a GLYPH_H x GLYPH_W float mask. Touching digits are split at the thinnest
    column near where a digit boundary should be."""
    m = red_mask(crop)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h >= MIN_H and a >= 20:
            blobs.append((x, (lab[y:y + h, x:x + w] == i).astype(np.float32)))
    out = []
    for _, g in sorted(blobs, key=lambda t: t[0]):
        parts = [g]
        while parts[-1].shape[1] > MAX_ONE_W:
            g2 = parts.pop()
            w = g2.shape[1]
            lo, hi = int(w * 0.35), max(int(w * 0.35) + 1, int(w * 0.65))
            if w > 2 * MAX_ONE_W:  # three or more: cut off one digit's width from the left
                lo, hi = 9, MAX_ONE_W
            cut = lo + int(np.argmin(g2[:, lo:hi].sum(axis=0)))
            parts += [g2[:, :cut], g2[:, cut:]]
        for p in parts:
            cols = np.where(p.sum(axis=0) > 0)[0]
            if len(cols):
                out.append(_place(p[:, cols[0]:cols[-1] + 1]))
    return out


@dataclass
class Reading:
    value: int | None   # None = not readable (no glyphs, or a glyph unlike every digit)
    worst: float        # largest glyph distance (0 = perfect match)
    n_glyphs: int


class DigitReader:
    """Nearest mean shape per digit, with a distance limit."""

    def __init__(self, templates: np.ndarray, digits: list[int], max_dist: float):
        self.t = templates.reshape(len(digits), -1).astype(np.float32)
        self.digits = list(digits)
        self.max_dist = max_dist

    @classmethod
    def load(cls, path: Path) -> DigitReader:
        z = np.load(path)
        return cls(z["templates"], [int(d) for d in z["digits"]], float(z["max_dist"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, templates=self.t.reshape(-1, GLYPH_H, GLYPH_W), digits=np.array(self.digits),
                 max_dist=self.max_dist)

    @classmethod
    def fit(cls, glyph_list: list[np.ndarray], digit_of: list[int], max_dist: float | None = None) -> DigitReader:
        """Mean glyph per digit. max_dist defaults to 1.5x the largest distance of a training glyph to its mean."""
        g = np.stack([x.reshape(-1) for x in glyph_list])
        d = np.array(digit_of)
        digits = sorted(set(d.tolist()))
        t = np.stack([g[d == k].mean(axis=0) for k in digits])
        if max_dist is None:
            own = np.linalg.norm(g - t[[digits.index(k) for k in d]], axis=1)
            max_dist = 1.5 * float(own.max())
        return cls(t, digits, max_dist)

    def classify(self, g: np.ndarray) -> tuple[int, float]:
        dist = np.linalg.norm(self.t - g.reshape(1, -1), axis=1)
        k = int(dist.argmin())
        return self.digits[k], float(dist[k])

    def read(self, crop: np.ndarray) -> Reading:
        gs = glyphs(crop)
        if not gs or len(gs) > 3:
            return Reading(None, float("inf"), len(gs))
        vals, worst = [], 0.0
        for g in gs:
            d, dist = self.classify(g)
            vals.append(d)
            worst = max(worst, dist)
        if worst > self.max_dist or (len(vals) > 1 and vals[0] == 0):
            return Reading(None, worst, len(gs))
        return Reading(int("".join(map(str, vals))), worst, len(gs))
