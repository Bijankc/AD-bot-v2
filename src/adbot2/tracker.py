"""Per-frame observations -> stable facts the decision rules (M2) may act on.

Every fact needs agreement over time, because single frames lie in known ways: white flashes look like loading
screens, the enemy count flickers 1 -> 0 -> 1 between waves, the raid clock text can lag ~5 s and OCR once dropped a
digit ("22:02" read as "2:02"). Nothing here sends input; it only says what is reliably on screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from adbot2.perception import Observation
from adbot2.smoothing import Agree, ScreenFilter

AUTO_OFF_FRAMES = 3       # confident OFF readings in a row before "Auto is off"
COUNT_AGREE = 2           # enemy count readings that must agree
CLEARED_HOLD_S = 4.0      # 0 enemies must hold this long ("the next wave came ~4 s after a 0 once")
CLOCK_TOLERANCE_S = 2     # two raid-clock readings must imply the same opening second within this
GF_AGREE = 2              # Group Finder banner readings in a row before "queued"
GF_CLEAR_S = 10.0         # banner gone this long before "not queued" (an open window dims it)


@dataclass
class Facts:
    screen: str | None = None          # None = unknown: wait, do not act
    auto_off: bool = False             # safe to press Auto
    enemy_count: int | None = None
    cleared: bool = False              # 0 enemies held for CLEARED_HOLD_S
    raid_opens_at: int | None = None   # second of the local hour this server's raid opens, once confirmed
    gf_queued: bool = False            # the Group Finder queue banner is on screen (debounced)


def local_time(t: float) -> float:
    """Epoch seconds shifted to local wall time, so that `local_time(t) % 3600` is the second of the local hour.
    Needed because time zones are not all whole hours (this PC is UTC+5:45: the raid at local :45 is UTC :00)."""
    return t + time.localtime(t).tm_gmtoff


class Tracker:
    def __init__(self, classes: list[str], half_life_s: float = 1.0):
        self.screens = ScreenFilter(classes, half_life_s=half_life_s)
        self._off_run = 0
        self.count = Agree(COUNT_AGREE)
        self.facts = Facts()
        self._zero_since: float | None = None
        self._clock_candidate: int | None = None
        self._gf_run = 0
        self._gf_last: float | None = None

    def update(self, o: Observation, t: float) -> Facts:
        """Feed one observation taken at time t (seconds since the epoch; the wall clock matters for the raid)."""
        f = self.facts
        if o.screen_probs is not None:
            f.screen, _ = self.screens.update(o.screen_probs, t)
        # Auto: OFF only after several confident OFF frames in a row, and only while the screen is a fight: menus and
        # windows without the AUTO button were the frames the reader came closest to calling OFF (18 Nov: P 0.92-0.97)
        self._off_run = self._off_run + 1 if (o.auto_is_off and f.screen == "fight") else 0
        f.auto_off = self._off_run >= AUTO_OFF_FRAMES
        # enemy count: agreeing readings; cleared once 0 has held
        if o.enemies:
            f.enemy_count = self.count.update(o.enemy_count)
        else:
            self.count.update(None)
            f.enemy_count = None
        if f.enemy_count == 0:
            self._zero_since = t if self._zero_since is None else self._zero_since
        else:
            self._zero_since = None
        f.cleared = self._zero_since is not None and t - self._zero_since >= CLEARED_HOLD_S
        # Group Finder queue: banner in GF_AGREE frames in a row; cleared only after it has been gone GF_CLEAR_S
        if o.gf_banner:
            self._gf_run += 1
            if self._gf_run >= GF_AGREE:
                f.gf_queued, self._gf_last = True, t
        elif o.gf_banner is not None:
            self._gf_run = 0
            if f.gf_queued and (self._gf_last is None or t - self._gf_last >= GF_CLEAR_S):
                f.gf_queued = False
        # raid clock: a reading implies an opening second; two that agree confirm it
        if o.menu_clock is not None:
            m, s = o.menu_clock
            ofs = int(round(local_time(t) + m * 60 + s)) % 3600
            cand = self._clock_candidate
            if cand is not None and min(abs(ofs - cand), 3600 - abs(ofs - cand)) <= CLOCK_TOLERANCE_S:
                f.raid_opens_at = cand
            self._clock_candidate = ofs
        return f
