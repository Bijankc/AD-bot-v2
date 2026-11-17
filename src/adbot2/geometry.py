"""Map any saved frame to the canonical game area: the old bot's 1920x1080 screen coordinates.

Roblox is per-monitor DPI aware and its top bar grows with the scale (58 px at 100%, 87 px at 150%), so the client is
1920x1017 at (0,23) at 100% and 1920x1046 at (0,-6) at 150%. Both end at screen row 1040, and the game lays out its UI
from the client bottom up, so game UI sits at the same canonical coordinates in both layouts (checked on real frames).

Canonical frame: 1920x1080, row y = screen row y of the old bot's layout. Only rows VALID_TOP..VALID_BOTTOM are game in
every layout (above: the title bar at 100%; below: whatever is behind the window), so everything outside is set to 0.
Models see `model_view()`: just that band, 1920x1017.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANVAS_W, CANVAS_H = 1920, 1080
CLIENT_BOTTOM = 1040          # screen row where the client ends, both layouts
VALID_TOP, VALID_BOTTOM = 23, 1040

# (dpi, client height) of known layouts. 1040 at 150% = the 1046 client clipped by the screen top (logger frames).
KNOWN = {(96, 1017): "100%", (144, 1046): "150%", (144, 1040): "150% clipped"}


class UnknownLayout(ValueError):
    pass


@dataclass
class GameArea:
    image: np.ndarray   # 1920x1080 BGR, 0 outside the valid band
    layout: str         # "100%", "150%", "150% clipped" or "screen" (an old-bot 1920x1080 screen grab)


def layout_of(width: int, height: int, dpi: int | None) -> str | None:
    if width != CANVAS_W:
        return None
    return KNOWN.get((dpi or 0, height))


def to_canvas(img: np.ndarray, dpi: int | None = None) -> GameArea:
    """A client-area capture (needs its DPI) or a 1920x1080 screen grab -> canonical game area.

    Raises UnknownLayout for anything else (window being resized, another monitor scale, a crop).
    """
    h, w = img.shape[:2]
    if dpi is None:
        if (w, h) != (CANVAS_W, CANVAS_H):
            raise UnknownLayout(f"{w}x{h} without a DPI is not a screen grab")
        out = img.copy()
        layout = "screen"
    else:
        layout = layout_of(w, h, dpi)
        if layout is None:
            raise UnknownLayout(f"client {w}x{h} at DPI {dpi}")
        out = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        top = CLIENT_BOTTOM - h  # 23 at 100%, -6 at 150%, 0 for the clipped 150% frames
        src = img[max(0, -top):]
        out[max(0, top):max(0, top) + len(src)] = src
    out[:VALID_TOP] = 0
    out[VALID_BOTTOM:] = 0
    return GameArea(out, layout)


def model_view(area: GameArea | np.ndarray) -> np.ndarray:
    """The part every layout shows: rows VALID_TOP..VALID_BOTTOM (1920x1017). A view, not a copy."""
    img = area.image if isinstance(area, GameArea) else area
    return img[VALID_TOP:VALID_BOTTOM]
