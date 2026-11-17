"""Weak screen labels from the old bot's log: what it believed was on screen, as time intervals.

The old bot logs a line when its state changes, not repeatedly, so a state line "holds" until the next one:

- hold:  the screen is shown from `settle` s after the line until `lead` s before the next state line (the screen
         changes before the bot notices), at most `max_s` long.
- at:    a one-off action ("pressing PLAY"): the screen is shown from `pre` s before the line to `post` s after.
- break: a state line with no reliable screen ("screen not recognised", "death: ..."); it only ends intervals.

Lines in INFO (UI timings, goal changes, OCR...) are not state lines and do not end intervals. Any other line does
(unknown = be careful). Where intervals with different screens overlap, there is no hint. These are hints for the
labeller and for measuring agreement, never ground truth.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from adbot2.oldlog import LogLine

F_AUTO, F_DIM = "fight_auto", "fight_dimension"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    screen: str | None
    kind: str            # hold | at | break
    a: float = 0.0       # hold: settle s after the line;  at: s before the line
    b: float = 0.0       # hold: lead s before the next state line;  at: s after the line
    max_s: float = 0.0   # hold: longest interval


def _r(name, pattern, screen, kind, a=0.0, b=0.0, max_s=0.0) -> Rule:
    return Rule(name, re.compile(pattern), screen, kind, a, b, max_s)


HOLD = dict(a=1.0, b=2.0)
RULES: list[Rule] = [
    # Auto fights (Raid, Speed Raid, Infinity)
    _r("raid_auto", r"^raid: Auto, waiting for the end", F_AUTO, "hold", **HOLD, max_s=600),
    _r("speed_auto", r"^Speed Raid: Auto, waiting for the end", F_AUTO, "hold", **HOLD, max_s=900),
    # An Infinity run on Auto is ~30 silent minutes; its results can show a few seconds before LEAVE.
    _r("inf_auto", r"^Infinity: (the game's Auto plays|ENEMIES LEFT did not change)", F_AUTO, "hold", a=1.0, b=5.0,
       max_s=2400),
    _r("not_dying", r"^enemies are not dying", F_AUTO, "hold", **HOLD, max_s=300),
    # Dimension fights
    _r("titan_fight", r"^Titan: (route leg|clearing after leg|leg \d+|fighting|boss|run begins|opening route|"
                      r"Shift Lock|reset|walking on|teleport gate|\S+ left and not coming|route done)", F_DIM, "hold",
       **HOLD, max_s=120),
    _r("dim_wait", r"^(Extreme Dimension \(Titan\)|Death Dimension): waiting for the results", F_DIM, "hold",
       **HOLD, max_s=60),
    # Waiting places
    _r("raid_closed", r"^(raid still closed, waiting|waiting until \d+:\d+, then Create)", "raid_world", "hold",
       a=1.0, b=2.0, max_s=600),
    _r("gf_queued", r"^Group Finder: (queued, waiting for the raid|raid open, waiting for the teleport)", "lobby",
       "hold", a=2.0, b=3.0, max_s=600),
    _r("party", r"^PARTY, Start in", "party", "hold", a=0.5, b=1.0, max_s=60),
    # Results
    _r("titan_results", r"^Titan: run over after \d+ s \(results\)", "results_dimension", "hold", a=0.3, b=0.5, max_s=6),
    _r("titan_results_ocr", r"^Titan results: ", "results_dimension", "at", 1.0, 0.3),
    _r("titan_retry", r"^Titan: (pressing RETRY|LEAVE \()", "results_dimension", "at", 1.5, 0.0),
    _r("inf_leave", r"^Infinity results: pressing LEAVE", "results_infinity", "at", 2.0, 0.0),
    _r("inf_ended", r"^Infinity ended by itself", "results_infinity", "hold", a=0.3, b=0.5, max_s=5),
    _r("rewards", r"^results: reading rewards", "results_raid", "hold", a=0.3, b=0.5, max_s=5),
    # Menus and windows (at the moment of a click or a find)
    _r("play", r"^pressing PLAY", "lobby", "at", 1.5, 0.0),
    _r("lobby_seen", r"^lobby \(PLAY visible\)", "lobby", "at", 1.0, 0.5),
    _r("modes_menu", r"^modes menu:", "modes_menu", "at", 1.0, 0.3),
    _r("raid_window", r"^(pressing Create \(raid\)|Create tab|Group Finder tab|looking for .+ Raid in the list|"
                      r"selecting .+ Raid$|enabling Friends Only|Friends Only)", "raid_window", "at", 1.0, 0.5),
    _r("dim_window", r"^(looking for .+ Dimension in the list|selecting .+ Dimension$|Titan: (pressing Create|"
                     r"choosing|Hardcore)|looking for Infinite Mode|selecting Infinite Mode|pressing Create \(infinity\))",
       "dimension_window", "at", 1.0, 0.5),  # Infinite Mode is picked in the Dimension window
    _r("bag", r"^loadout \d+ -> \d+", "bag", "at", 1.0, 0.3),
    _r("speed_window", r"^Speed Raid: (looking for a free character|start \(character)", "speed_raid_window", "at",
       1.0, 0.0),
    _r("restart", r"^(restarting Roblox|launching private server link)", "loading", "at", -3.0, 10.0),
    # State changes with no reliable screen
    _r("break", r"^(waiting \(screen not recognised\)|screen not recognised|death: |Roblox is not responding|stopped|"
                r"start v|RESTART|in a run \(AUTO|entering Dimension|teleporting|lobby loaded|leaving Infinity|"
                r"pressing Start)", None, "break"),  # "in a run (AUTO" was logged on a Create window (false AUTO)
]

# Not state lines: they never end an interval.
INFO = re.compile(
    r"^(ui: |screenshot:|OCR server|goal: |setting |cooldown |Infinity in \d+ s:|raid: (Auto pressed|Auto looks off|"
    r"in-fight)|Speed Raid: Auto pressed|Infinity: Auto( pressed|:)|Auto: (on|off)|display |next Speed Raid|"
    r"\d+ \d+ times in a row|.+matched a blob|setup: |Rewards |RUN SUMMARY|session summary|Titan: (gems|count \d+ read|"
    r"leg \d+ walks|leg \d+: boost)|.+: fight lasted|.+: (gems|tokens) |tokens: could not read|Speed Raid: (weekly|gems)|"
    r"recording has|raid clock|LEAVE at |lobby check:|Extreme: today|Extreme window text|Speed Raid is available|"
    r"Speed Raid not available|Group Finder: queued for|loadout \d+ (✓|\?)|loadout \d+: opening Bag)"
)


def match_rule(text: str) -> Rule | None:
    for r in RULES:
        if r.pattern.search(text):
            return r
    return None


def is_state_line(text: str) -> bool:
    return match_rule(text) is not None or not INFO.search(text)


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime
    screen: str
    rule: str


def intervals(lines: list[LogLine]) -> list[Interval]:
    """Screen intervals from log lines (any order; sorted here)."""
    lines = sorted(lines, key=lambda ln: ln.time)
    states = [(ln, match_rule(ln.text)) for ln in lines if is_state_line(ln.text)]
    out: list[Interval] = []
    for i, (ln, r) in enumerate(states):
        if r is None or r.screen is None:
            continue
        if r.kind == "at":
            out.append(Interval(ln.time - timedelta(seconds=r.a), ln.time + timedelta(seconds=r.b), r.screen, r.name))
            continue
        prev = states[i - 1] if i else None
        if r.name == "raid_auto" and prev and prev[1] and prev[1].name == "rewards"                 and ln.time - prev[0].time < timedelta(seconds=30):
            continue  # logged again on the results screen after the rewards were read
        nxt = states[i + 1] if i + 1 < len(states) else None
        # settle/lead only where the screen changes: back-to-back lines of one fight join up
        same_prev = prev is not None and prev[1] is not None and prev[1].screen == r.screen
        same_next = nxt is not None and nxt[1] is not None and nxt[1].screen == r.screen
        start = ln.time + timedelta(seconds=0 if same_prev else r.a)
        stop = nxt[0].time if nxt else ln.time + timedelta(seconds=r.max_s)
        end = min(stop - timedelta(seconds=0 if same_next else r.b), ln.time + timedelta(seconds=r.max_s))
        if end > start:
            out.append(Interval(start, end, r.screen, r.name))
    out.sort(key=lambda iv: iv.start)
    return out


class HintIndex:
    """Look up the hinted screen at a moment. Conflicting overlaps give no hint."""

    def __init__(self, ivs: list[Interval]):
        self.ivs = sorted(ivs, key=lambda iv: iv.start)
        self.starts = [iv.start for iv in self.ivs]
        longest = max((iv.end - iv.start for iv in self.ivs), default=timedelta(0))
        self._span = longest

    def at(self, t: datetime) -> tuple[str | None, str | None]:
        """(screen, rule) or (None, None)."""
        lo = bisect.bisect_left(self.starts, t - self._span)
        hi = bisect.bisect_right(self.starts, t)
        hits = [iv for iv in self.ivs[lo:hi] if iv.start <= t <= iv.end]
        screens = {iv.screen for iv in hits}
        if len(screens) != 1:
            return None, None
        return hits[-1].screen, hits[-1].rule


def context_at(lines: list[LogLine], t: datetime, before: int = 3, after: int = 1) -> list[str]:
    """A few state lines around t, with times, for the labelling page."""
    times = [ln.time for ln in lines]
    k = bisect.bisect_right(times, t)
    picked = [ln for ln in lines[max(0, k - 12):k] if is_state_line(ln.text)][-before:]
    picked += [ln for ln in lines[k:k + 12] if is_state_line(ln.text)][:after]
    return [f"{ln.time:%H:%M:%S}{' >' if ln.time > t else ' '} {ln.text}" for ln in picked]
