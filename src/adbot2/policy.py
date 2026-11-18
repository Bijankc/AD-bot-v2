"""What the bot would do next, from stable facts (adbot2.tracker). Decisions only: nothing here sends input.

First slice (M2): the Infinity loop and the hourly schedule around the raid, ported as plain rules from the old bot's
spec (..\\AD-bot\\CLAUDE.md, "How the bot decides"):

- Priority raid > Speed Raid > Infinity. Times are "raid time": seconds since this server's raid opened (0 = opens),
  so the same rules work for any server once the raid clock is confirmed.
- Infinity runs until INF_STOP (57:30 raid time, 2.5 min before the raid), then reset (Esc, R, Enter) and LEAVE.
- In the lobby within RAID_SOON of INF_STOP no new Infinity run starts.
- Auto is pressed when the tracker says it is off during a fight (the game turns it off after a teleport).
- Unknown screen = wait. Screens this slice does not handle yet give Action("unhandled", ...), never a guess.

Actions are names with a reason; the input layer (later, --live only) turns them into clicks and keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from adbot2.tracker import Facts

INF_STOP = 57 * 60 + 30     # raid time to leave Infinity (real :42:30 when the raid opens at :45:00)
RAID_SOON = 360             # in the lobby this close to INF_STOP: do not start Infinity
CREATE_FROM = 59 * 60 + 30  # raid time to start creating the raid


@dataclass(frozen=True)
class Action:
    name: str      # wait | press_play | open_dimension | pick_infinite | create | start | press_auto | reset |
    #                leave | retry | unhandled
    why: str


@dataclass
class Settings:
    raid: bool = True
    speed: bool = True
    infinity: bool = True


def raid_time(now: float, opens_at: int | None) -> int | None:
    """Seconds since the raid opened, in [0, 3600), or None while the raid clock is unknown."""
    if opens_at is None:
        return None
    return int(now - opens_at) % 3600


def goal(now: float, facts: Facts, s: Settings) -> str:
    """raid | infinity (Speed Raid is not modelled in this slice)."""
    rt = raid_time(now, facts.raid_opens_at)
    if s.raid and rt is not None and (rt >= INF_STOP or rt < 10 * 60):
        return "raid"     # from 2.5 min before opening until creation closes (~10 min after)
    return "infinity" if s.infinity else "raid"


def decide(now: float, facts: Facts, s: Settings | None = None, mode: str | None = None) -> Action:
    """One decision. `mode` = what the current run was entered as ("infinity", ...), None if not known: reset only
    ever leaves a run entered as Infinity (a raid or Speed Raid fight near the raid hour must not be left)."""
    s = s or Settings()
    g = goal(now, facts, s)
    rt = raid_time(now, facts.raid_opens_at)
    scr = facts.screen
    if scr is None:
        return Action("wait", "screen unknown")
    if scr in ("loading", "respawn", "dead"):
        return Action("wait", scr)
    if scr == "fight":
        if facts.auto_off:
            return Action("press_auto", "Auto is off in a fight")
        if mode == "infinity" and g == "raid" and rt is not None and INF_STOP <= rt < CREATE_FROM:
            return Action("reset", "the raid opens soon: leave the Infinity run (Esc, R, Enter)")
        return Action("wait", "fighting on Auto")
    if scr == "results_infinity":
        return Action("leave", "Infinity results")
    if scr == "lobby":
        if g == "infinity" and rt is not None and INF_STOP - RAID_SOON <= rt < INF_STOP:
            return Action("unhandled", "raid soon: go and wait for the raid (not in this slice)")
        if g == "infinity":
            return Action("press_play", "lobby, goal Infinity")
        return Action("unhandled", "lobby, goal raid (not in this slice)")
    if scr == "modes_menu":
        if g == "infinity":
            return Action("open_dimension", "modes menu, goal Infinity")
        return Action("unhandled", "modes menu, goal raid (not in this slice)")
    if scr == "dimension_window":
        if g == "infinity":
            return Action("pick_infinite", "Dimension window: Infinite Mode, then Create")
        return Action("unhandled", "Dimension window while the goal is the raid")
    if scr == "party":
        return Action("start", "PARTY: Start")
    return Action("unhandled", f"screen {scr} not in this slice")


class Policy:
    """decide() plus the memory of which mode the current run was entered as."""

    def __init__(self, settings: Settings | None = None, own_mode: bool = True):
        self.s = settings or Settings()
        self.mode: str | None = None
        self.own_mode = own_mode   # False in shadow mode: the mode is set from what the old bot actually entered

    def step(self, now: float, facts: Facts) -> Action:
        if self.own_mode and facts.screen in ("lobby", "modes_menu", "raid_world"):
            self.mode = None
        a = decide(now, facts, self.s, self.mode)
        if self.own_mode and a.name == "pick_infinite":
            self.mode = "infinity"
        return a
