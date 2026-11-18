"""Read one line of game text from a fixed band: the raid clock and similar short fields.

Game text is a fixed bold font drawn on a dark, mostly opaque bar, so it is read like the enemy count (digits.py):
text pixels -> glyphs (parts stacked vertically, such as a colon or the dot of an "i", are merged; glyphs too wide
for one character are split) -> nearest named shape -> a string, then a pattern pulls the value out. A glyph unlike
every shape becomes "?", and a value is only returned when the pattern matches cleanly, so a dropped digit
("22:02" seen as "2:02", the old bot's worst OCR error) cannot pass: it needs a "?" -> no reading.

Fields are in the canonical frame (centre- and bottom-anchored UI, same box at 100% and 150%). Shapes are named once
from the user's frames (tools/text_templates.py) and live under <ADBOT2_HOME>/models/text/ (never committed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TextField:
    name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the canonical frame
    min_bright: int = 170            # text pixels: max channel above this ...
    max_dark_bg: int = 110           # ... the bar under the text is darker than this (checked on the band's median)
    max_w: int = 15                  # wider glyphs are split (touching characters)
    canvas: tuple[int, int] = (24, 16)  # (h, w) a glyph is placed on (rows as in the band)
    pattern: str = r"(\d{1,2}):(\d\d)$"
    help: str = ""


FIELDS: dict[str, TextField] = {f.name: f for f in [
    # modes menu, RAID card: "Raid Opens in: 04:21" (beige on a dark bar, centred, always MM:SS)
    TextField("menu_clock", (700, 881, 840, 902), max_w=12, pattern=r"(\d\d):(\d\d)$", help="RAID card countdown in the modes menu"),
    # Group Finder queue banner: "<n>/8 Raid opens in 0:33" (shifts with the player count) or "Starting soon..."
    TextField("gf_clock", (850, 128, 1010, 152), help="Group Finder banner countdown"),
]}


def cut_field(img: np.ndarray, f: TextField) -> np.ndarray:
    x0, y0, x1, y1 = f.box
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def text_mask(crop: np.ndarray, f: TextField) -> np.ndarray:
    return (crop.max(axis=2) > f.min_bright).astype(np.uint8)


def on_dark_bar(crop: np.ndarray, f: TextField) -> bool:
    """Text on the plain dark bar: a few % text pixels, and everything that is not text (or its anti-aliased edge)
    is dark and even. Bright scenery behind a see-through band fails this."""
    v = crop.max(axis=2)
    m = text_mask(crop, f)
    frac = float(m.mean())
    if not 0.03 <= frac <= 0.4:
        return False
    edge = cv2.dilate(m, np.ones((3, 3), np.uint8))
    bg = v[edge == 0]
    return bg.size > 0 and float(np.percentile(bg, 95)) < f.max_dark_bg and float(bg.std()) < 18


def segment(crop: np.ndarray, f: TextField) -> list[tuple[int, np.ndarray]]:
    """[(x, glyph canvas)] left to right. Small dots (colon, i-dot) join the character they overlap in x or stack
    into a colon; glyphs wider than one character are split. A glyph keeps its rows within the band (fixed field)."""
    m = text_mask(crop, f)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 4)
    comps = sorted((int(st[i, 0]), int(st[i, 0] + st[i, 2]), i, int(st[i, 2]) <= 4 and int(st[i, 3]) <= 5)
                   for i in range(1, n) if st[i, 4] >= 3)
    groups: list[list] = []   # [x0, x1, members, all_small]
    for x0, x1, i, small in comps:
        g = groups[-1] if groups else None
        if g is not None and (small or g[3]) and x0 < g[1] - 1:
            g[1] = max(g[1], x1)
            g[2].append(i)
            g[3] = g[3] and small
        else:
            groups.append([x0, x1, [i], small])
    ch, cw = f.canvas
    out = []
    for x0, x1, members, _ in groups:
        g = np.isin(lab[:, x0:x1], members).astype(np.float32)
        parts = [(x0, g)]
        while parts[-1][1].shape[1] > f.max_w:
            px, pg = parts.pop()
            w = pg.shape[1]
            lo, hi = max(3, int(w * 0.3)), max(4, min(w - 3, int(w * 0.7)))
            cut = lo + int(np.argmin(pg[:, lo:hi].sum(axis=0)))
            parts += [(px, pg[:, :cut]), (px + cut, pg[:, cut:])]
        for px, pg in parts:
            cols = np.where(pg.any(axis=0))[0]
            if len(cols) == 0:
                continue
            pg = pg[:, cols[0]:cols[-1] + 1]
            c = np.zeros((ch, cw), np.float32)
            h, w = min(pg.shape[0], ch), min(pg.shape[1], cw)
            c[:h, :w] = pg[:h, :w]
            out.append((px + int(cols[0]), c))
    return out


@dataclass
class TextReading:
    text: str               # the glyphs' texts ("?" = unlike every shape; a template may stand for "2:")
    value: tuple | None     # pattern groups as ints, or None


class TextReader:
    """Nearest named shape per glyph, with a distance limit."""

    def __init__(self, field: TextField, templates: np.ndarray, chars: list[str], max_dist: float):
        self.f = field
        self.t = templates.reshape(len(chars), -1).astype(np.float32)
        self.chars = list(chars)
        self.max_dist = max_dist
        self.rx = re.compile(field.pattern)

    @classmethod
    def from_examples(cls, field: TextField, glyph_list: list[np.ndarray], chars: list[str],
                      max_dist: float) -> TextReader:
        """Nearest example (not a mean): keeps similar small digits (8 / 9 / 6) apart."""
        return cls(field, np.stack(glyph_list), list(chars), max_dist)

    @classmethod
    def fit(cls, field: TextField, glyph_list: list[np.ndarray], chars: list[str],
            max_dist: float | None = None) -> TextReader:
        """One mean shape per text given; max_dist defaults to 1.5x the 99.5th percentile of the training glyphs'
        distances to their own mean."""
        g = np.stack([x.reshape(-1) for x in glyph_list])
        keys = sorted(set(chars))
        means = np.stack([g[[c == k for c in chars]].mean(axis=0) for k in keys])
        if max_dist is None:
            own = np.linalg.norm(g - means[[keys.index(c) for c in chars]], axis=1)
            max_dist = 1.5 * float(np.percentile(own, 99.5))  # a stray member must not loosen the limit
        return cls(field, means, keys, max_dist)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, field=self.f.name, templates=self.t.reshape(-1, *self.f.canvas), chars=np.array(self.chars),
                 max_dist=self.max_dist)

    @classmethod
    def load(cls, path: Path) -> TextReader:
        z = np.load(path)
        return cls(FIELDS[str(z["field"])], z["templates"], [str(c) for c in z["chars"]], float(z["max_dist"]))

    def read_crop(self, crop: np.ndarray) -> TextReading:
        chars = []
        for _, g in segment(crop, self.f):
            d = np.linalg.norm(self.t - g.reshape(1, -1), axis=1)
            k = int(d.argmin())
            chars.append(self.chars[k] if d[k] <= self.max_dist else "?")
        text = "".join(chars)
        m = self.rx.search(text)
        return TextReading(text, tuple(int(x) for x in m.groups()) if m else None)

    def read(self, frame: np.ndarray) -> TextReading:
        return self.read_crop(cut_field(frame, self.f))
