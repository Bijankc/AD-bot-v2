"""Fixed buttons read by colour, no model: is this button on screen right now?

The PARTY window (after Create in the Dimension or raid window, before the run) has an orange "Start" button with
white text. Labels put that window under dimension_window (it is the same frame, the window's content changes), so
the screen class alone cannot tell "pick a mode, Create" from "press Start". The window is centre-anchored, so one
canonical box serves both layouts (20 Nov: 10 PARTY frames, 100% and 150%, button within a few px).

In the Create view the same spot holds the dark "Back" button, so "mostly orange with some white" separates them.
Thresholds measured on 21 Nov with tools/button_check.py (10 Start frames, 434 window frames, 600 others): all Start
frames found, the only other hits are PARTY frames outside the 1.5 s window. The policy only asks on window screens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

START_BOX = (1078, 703, 1228, 757)   # x0, y0, x1, y1 in the canonical frame: inside the button, rounded corners off


@dataclass(frozen=True)
class ButtonRead:
    orange: float   # share of orange-gradient pixels (button face)
    white: float    # share of white pixels (text)

    @property
    def shown(self) -> bool:
        return self.orange >= ORANGE_MIN and WHITE_MIN <= self.white <= WHITE_MAX


ORANGE_MIN = 0.78   # 21 Nov: Start frames 0.86-0.90; Weekly Quests "Complete" buttons overlap at 0.60-0.62
WHITE_MIN = 0.04
WHITE_MAX = 0.35


def read_start(img: np.ndarray) -> ButtonRead:
    """`img`: canonical 1920x1080 BGR frame (GameArea.image)."""
    x0, y0, x1, y1 = START_BOX
    c = img[y0:y1, x0:x1].astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    orange = (r >= 200) & (g >= 60) & (g <= 175) & (b <= 130) & (r - g >= 50)
    white = (r >= 225) & (g >= 225) & (b >= 225)
    n = orange.size
    return ButtonRead(float(orange.sum()) / n, float(white.sum()) / n)
