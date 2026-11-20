"""Shadow mode helpers: what v2 copies from the old bot while it only watches.

In shadow mode v2 decides but does not act, so the things it would normally remember from its own actions (which run
it entered, the loadout it set, when the next Speed Raid is due, which modes are switched on) must come from what the
old bot really did. OldBotState reads that from the old bot's log lines, fed in time order: all at once for stored
frames (tools/shadow.py), or a few new lines at a time from raid.log while live (tools/shadow_live.py).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

from adbot2.oldlog import LogLine, parse_line
from adbot2.policy import Policy, Settings

# what the old bot entered, for the policy's mode memory; None = no run any more
ENTERED = [
    (re.compile(r"^pressing Create \(infinity\)"), "infinity"),
    (re.compile(r"^Titan: pressing Create"), "titan"),
    (re.compile(r"^Speed Raid: start \(character"), "speed"),
    (re.compile(r"^(pressing Create \(raid\)|Group Finder: (raid open|queued for )|raid: Auto)"), "raid"),
    (re.compile(r"^(pressing PLAY|lobby \(PLAY visible\)|restarting Roblox)"), None),
]
LOADOUT_LINE = re.compile(r"^loadout (\d): opening Bag")   # the intent: "loadout N ✓" comes with PLAY
SPEED_AT = re.compile(r"next Speed Raid at (\d\d):(\d\d)")
SPEED_NOW = re.compile(r"\(Speed Raid is available\)")
START_LINE = re.compile(r"^start v.*loadouts=(\d), raid=(\d), infinity=(\d), speed=(\d), raidEntry=(\w+)")


def speed_due(ln: LogLine) -> float | None:
    """When the old bot planned its next Speed Raid, from its goal lines."""
    if SPEED_NOW.search(ln.text):
        return ln.time.timestamp()
    if m := SPEED_AT.search(ln.text):
        due = ln.time.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if due < ln.time - timedelta(hours=12):
            due += timedelta(days=1)
        return due.timestamp()
    return None


def settings(ln: LogLine) -> Settings | None:
    if m := START_LINE.search(ln.text):
        lo, raid, inf, speed, entry = m.groups()
        # infinity=0 was the Titan farming days (titan is not in the start line)
        return Settings(raid=raid == "1", speed=speed == "1", infinity=inf == "1", dimension=inf == "0",
                        raid_entry=entry, loadouts=lo == "1")
    return None


class OldBotState:
    """The old bot's mode, loadout, Speed Raid schedule and settings, from its log lines in time order."""

    def __init__(self):
        self.mode: str | None = None
        self.loadout: int | None = None
        self.speed_due: float | None = None
        self.settings = Settings()
        self.last: datetime | None = None

    def feed(self, ln: LogLine) -> None:
        self.last = ln.time
        for rx, mode in ENTERED:
            if rx.search(ln.text):
                self.mode = mode
                break
        if m := LOADOUT_LINE.search(ln.text):
            self.loadout = int(m.group(1))
        if (due := speed_due(ln)) is not None:
            self.speed_due = due
        if (s := settings(ln)) is not None:
            self.settings = s

    def apply(self, pol: Policy) -> None:
        pol.mode = self.mode
        pol.m.loadout = self.loadout
        pol.m.speed_due = self.speed_due
        pol.s = self.settings


class LogFollower:
    """New lines of a growing log file, read with a quick open and close each time (a held handle once blocked the
    old bot's writes for hours). Starts at the end; a file that shrank (rotated at midnight) is read from the top."""

    def __init__(self, path: Path, from_start: bool = False):
        self.path = path
        self.pos = 0 if from_start or not path.exists() else path.stat().st_size
        self._rest = b""

    def new_lines(self) -> list[LogLine]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:
            self.pos, self._rest = 0, b""
        if size == self.pos:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(size - self.pos)
        self.pos += len(data)
        data = self._rest + data
        *full, self._rest = data.split(b"\n")
        year = datetime.now().year
        out = []
        for raw in full:
            ln = parse_line(raw.decode("utf-8", errors="replace"), year)
            if ln is not None:
                out.append(ln)
        return out


GAP_S = 30.0          # frames further apart than this (Roblox not in front) start the tracker again
ROW_EVERY_S = 30.0    # a decision row at least this often even when nothing changes


class LiveShadow:
    """Tracker + policy fed with live observations; the policy's run memory follows the old bot's log."""

    def __init__(self, classes: list[str], opens: int | None, follower: LogFollower | None = None):
        from adbot2.tracker import Tracker

        self._new_tracker = lambda: Tracker(classes)
        self.tracker = self._new_tracker()
        self.tracker.facts.raid_opens_at = opens
        self.pol = Policy(own_mode=False)
        self.old = OldBotState()
        self.follower = follower
        self._prev_t: float | None = None
        self._last_row: tuple[str, float] | None = None

    def read_log(self) -> int:
        lines = self.follower.new_lines() if self.follower else []
        for ln in lines:
            self.old.feed(ln)
        return len(lines)

    def step(self, o, t: float):
        """(facts, action, write): write is True when the decision changed or ROW_EVERY_S passed."""
        if self._prev_t is not None and t - self._prev_t > GAP_S:
            opens = self.tracker.facts.raid_opens_at
            self.tracker = self._new_tracker()
            self.tracker.facts.raid_opens_at = opens
        self._prev_t = t
        f = self.tracker.update(o, t)
        self.old.apply(self.pol)
        a = self.pol.step(t, f)
        write = self._last_row is None or a.name != self._last_row[0] or t - self._last_row[1] >= ROW_EVERY_S
        if write:
            self._last_row = (a.name, t)
        return f, a, write
