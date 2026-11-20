"""What the bot would do next, from stable facts (adbot2.tracker). Decisions only: nothing here sends input.

Rules ported from the old bot's spec (..\\AD-bot\\CLAUDE.md, "How the bot decides", and its PickTarget / SpeedReady /
SpeedWait / Group Finder code). Slices so far: Infinity, the hourly raid (Group Finder queue with the Create fallback,
or Create), Speed Raid, and the schedule between them. Titan and other Dimensions are not in yet ("unhandled").

- Times are "raid time": seconds since this server's raid opened (0 = opens), so the rules work for any server once
  the raid clock is confirmed. A "raid hour" runs from INF_STOP to the next INF_STOP and is named by the raid in it.
- Priority raid > Speed Raid > Infinity:
  raid from INF_STOP (57:30) until it opens; outside a fight also when it is <= RAID_SOON from INF_STOP; and in the
  first RAID_WINDOW (10 min) after it opens while no raid fight has started this raid hour (the game allows one).
  Speed Raid when due (30 min after the last one started, or 30:15 after its fight began once it ended) and, with
  the raid on, not after minute SPEED_LAST_MIN. Then a Dimension (if on), then Infinity.
- Infinity is left (reset: Esc, R, Enter, then LEAVE) when the goal becomes the raid, or Speed Raid unless
  SpeedWait says it is not worth it (raid near, or the run is 30+ min old and ends soon by itself).
- Group Finder: queue in the raid world; still queued GF_FALLBACK (120 s) after the raid opened -> cancel and create
  the raid this hour instead. The raid world has no menu way back to the lobby: with another goal Roblox restarts.
- Watchdog (old bot's restart rules): Speed Raid fight over 12 min (3 min when the raid is due), raid fight over
  15 min, raid world without a raid goal for 90 s, disconnected, the same picture for 3 min (not while queued),
  screen unknown 90 s (Esc once) then 180 s, and a preventive restart of a Roblox running 2 h, in the lobby 7-50 min
  into the raid hour. Restarts keep a gap (10 min; 60 s for lost connections and stuck runs). A raid fight cut off
  by a restart in its first 10 min, before its results, may be tried once more that raid hour.
- Unknown screen = wait. Screens not handled yet give Action("unhandled", ...), never a guess.

Actions are names with a reason; the input layer (later, --live only) turns them into clicks and keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adbot2.tracker import Facts, local_time

INF_STOP = 57 * 60 + 30     # raid time to leave Infinity (real :42:30 when the raid opens at :45:00)
RAID_SOON = 360             # out of a fight this close to INF_STOP: go and wait for the raid
CREATE_FROM = 59 * 60 + 30  # raid time to start pressing Create (it fails until the raid opens)
RAID_WINDOW = 10 * 60       # a raid can only be created in about the first 10 min
GF_FALLBACK = 120           # still queued this long after the raid opened: cancel and create it
SPEED_EVERY = 30 * 60
SPEED_LAST_MIN = 52         # with the raid on, no Speed Raid starts after this minute of the raid hour
SPEED_MAX = 12 * 60         # a Speed Raid fight longer than this is left (Roblox restart)
SPEED_WAIT_MAX = 12 * 60    # a due Speed Raid is put off at most this long
INF_LONG = 30 * 60          # an Infinity run this old ends soon by itself
RAID_MAX = 15 * 60          # a raid still a fight after this long restarts Roblox (10 Nov: 5 h behind a dialog)
SPEED_RAID_DUE_MAX = 180    # a Speed Raid fight this long is left when the raid is due (the raid is worth more)
RAID_WORLD_GRACE = 90       # raid world without a raid goal this long: restart (no menu way back)
ROBLOX_MAX_AGE = 120 * 60   # Roblox running this long is restarted in the lobby between raids (froze after 3.5 h)
PREVENT_FROM, PREVENT_TO = 420, 3000   # raid time window for the preventive restart (after the raid, far from next)
UNKNOWN_ESC = 90            # screen unknown this long: press Esc once (a stuck menu)
UNKNOWN_RESTART = 180       # still unknown: restart
FROZEN_S = 180              # the same picture this long (a frozen render): restart, not while queued
RESTART_GAP = 600           # restarts at most every 10 min ...
RESTART_GAP_FAST = 60       # ... or every minute for lost connections and stuck runs
LOADOUT = {"raid": 1, "infinity": 2, "speed": 3, "dimension": 2}

IN_RUN = ("fight", "respawn", "dead", "revive")
RUN_OVER = ("lobby", "modes_menu", "raid_world", "results_raid", "results_infinity", "results_dimension")


@dataclass(frozen=True)
class Action:
    name: str      # wait | set_loadout | press_play | open_dimension | pick_infinite | open_raid | queue_gf |
    #                cancel_queue | create_raid | open_speed | speed_start | start | press_auto | reset | leave |
    #                close_window | press_esc | restart_roblox | unhandled
    why: str
    arg: int | None = None   # set_loadout: the loadout number


@dataclass
class Settings:
    raid: bool = True
    speed: bool = True
    infinity: bool = True
    dimension: bool = False  # Titan and other Dimensions (before Infinity; choosing one is not in a slice yet)
    raid_entry: str = "gf"   # gf (Group Finder queue, Create as fallback) | create
    loadouts: bool = True    # Bag loadouts: 1 raid, 2 Infinity, 3 Speed Raid


@dataclass
class Memory:
    """What the bot has to remember between frames (the screen alone does not say it)."""
    mode: str | None = None            # what the current run was entered as: infinity | raid | speed | titan
    fight_since: float | None = None   # when the current run's fight began
    raid_done: int | None = None       # raid hour in which a raid fight started
    create_hour: int | None = None     # raid hour in which Group Finder gave up: create the raid instead
    speed_due: float | None = None     # next Speed Raid allowed from (None = now)
    loadout: int | None = None         # loadout in use, None = not known
    screen: str | None = None          # screen fact at the last step, and since when
    screen_since: float | None = None
    esc_done: bool = False             # Esc pressed for this unknown spell
    last_restart: float | None = None
    roblox_started: float | None = None   # set by whoever starts Roblox (the input layer); None = not known
    raid_end_seen: int | None = None   # raid hour whose raid results were seen (no retry after that)
    raid_retry: int | None = None      # raid hour already retried once after a cut-off raid


def raid_time(now: float, opens_at: int | None) -> int | None:
    """Seconds since the raid opened, in [0, 3600), or None while the raid clock is unknown. `opens_at` is the
    second of the local hour (the old bot's and the game's clock)."""
    if opens_at is None:
        return None
    return int(local_time(now) - opens_at) % 3600


def raid_hour(now: float, opens_at: int) -> int:
    """Number of the raid hour (INF_STOP to INF_STOP), named by the raid that opens in it."""
    return math.floor((local_time(now) - opens_at + 3600 - INF_STOP) / 3600)


def speed_ready(now: float, rt: int | None, s: Settings, m: Memory) -> bool:
    if not s.speed or (m.speed_due is not None and now < m.speed_due):
        return False
    if not s.raid:
        return True
    return rt is not None and rt // 60 < SPEED_LAST_MIN


def speed_wait(now: float, rt: int | None, s: Settings, m: Memory) -> bool:
    """Speed Raid is due but leaving the Infinity run for it is not worth it (old bot's SpeedWait)."""
    if m.speed_due is not None and now - m.speed_due > SPEED_WAIT_MAX:
        return False
    if s.raid and rt is not None and INF_STOP - rt <= 10 * 60:
        return True    # the raid is near: Infinity is left at INF_STOP anyway, Speed Raid comes after the raid
    return m.fight_since is not None and now - m.fight_since > INF_LONG


def watchdog(now: float, facts: Facts, g: str, rt: int | None, m: Memory) -> tuple[Action, float] | None:
    """A restart (or Esc) the situation calls for, with the gap it needs since the last restart; None if all is well."""
    scr = facts.screen
    held = now - m.screen_since if m.screen_since is not None else 0.0
    if scr is None and held >= UNKNOWN_RESTART:
        return Action("restart_roblox", f"screen not recognised for {held / 60:.0f} min"), RESTART_GAP
    if scr is None and held >= UNKNOWN_ESC and not m.esc_done:
        return Action("press_esc", "screen not recognised for 90 s: Esc once (a stuck menu)"), 0
    if scr == "disconnected":
        return Action("restart_roblox", "Roblox lost the connection"), RESTART_GAP_FAST
    if facts.still_since is not None and now - facts.still_since >= FROZEN_S and not facts.gf_queued:
        return Action("restart_roblox", "the game picture has not changed for 3 min"), 0
    if m.fight_since is not None and scr in IN_RUN:
        run = now - m.fight_since
        if m.mode == "speed" and (run > SPEED_MAX or g == "raid" and run > SPEED_RAID_DUE_MAX):
            return Action("restart_roblox", "Speed Raid takes too long"), RESTART_GAP_FAST
        if m.mode == "raid" and run > RAID_MAX:
            return Action("restart_roblox", "raid takes too long"), RESTART_GAP_FAST
    if scr == "raid_world" and g != "raid" and held >= RAID_WORLD_GRACE:
        return Action("restart_roblox", "in the raid world without a raid (no menu way back)"), RESTART_GAP_FAST
    if (scr == "lobby" and g != "raid" and rt is not None and PREVENT_FROM <= rt < PREVENT_TO
            and m.roblox_started is not None and now - m.roblox_started >= ROBLOX_MAX_AGE):
        age = int(now - m.roblox_started) // 60
        return Action("restart_roblox", f"preventive, it has been running {age // 60} h {age % 60} min"), RESTART_GAP
    return None


def goal(now: float, facts: Facts, s: Settings, m: Memory | None = None) -> str:
    """raid | speed | dimension | infinity | none."""
    m = m or Memory()
    rt = raid_time(now, facts.raid_opens_at)
    if s.raid and rt is not None:
        if rt >= INF_STOP:
            return "raid"
        in_run = m.fight_since is not None or facts.screen in IN_RUN
        if not in_run and 0 < INF_STOP - rt <= RAID_SOON:
            return "raid"
        if rt < RAID_WINDOW and m.raid_done != raid_hour(now, facts.raid_opens_at):
            return "raid"
    if speed_ready(now, rt, s, m):
        return "speed"
    if s.dimension:
        return "dimension"
    if s.infinity:
        return "infinity"
    return "raid" if s.raid else "none"


def decide(now: float, facts: Facts, s: Settings | None = None, m: Memory | None = None) -> Action:
    """One decision. Reset only ever leaves a run entered as Infinity: a raid or Speed Raid fight near the raid hour
    must not be left."""
    s = s or Settings()
    m = m or Memory()
    g = goal(now, facts, s, m)
    rt = raid_time(now, facts.raid_opens_at)
    scr = facts.screen
    if (w := watchdog(now, facts, g, rt, m)) is not None:
        a, gap = w
        if a.name != "restart_roblox" or m.last_restart is None or now - m.last_restart >= gap:
            return a
    if scr is None:
        return Action("wait", "screen unknown")
    if scr in ("loading", "respawn", "dead"):
        return Action("wait", scr)
    if scr == "revive":
        return Action("wait", "REVIVE offered: never pressed (costs gems)")
    if scr == "disconnected":
        return Action("wait", "disconnected: restart gap")
    if scr == "fight":
        if facts.auto_off:
            return Action("press_auto", "Auto is off in a fight")
        if m.mode == "infinity" and (g == "raid" or g == "speed" and not speed_wait(now, rt, s, m)):
            return Action("reset", f"goal {g}: leave the Infinity run (Esc, R, Enter)")
        return Action("wait", "fighting on Auto")
    if scr == "results_infinity":
        return Action("leave", "Infinity results")
    if scr in ("results_raid", "results_dimension") and m.mode == "speed":
        return Action("leave", "Speed Raid results: LEAVE")   # rank diamond, REWARDS, only LEAVE (like a Dimension's)
    if scr == "results_raid":
        return Action("wait", "raid results (a Group Finder raid returns to the lobby by itself)")
    if g == "none":
        return Action("wait", "nothing to farm")
    if scr == "lobby":
        want = LOADOUT[g]
        if s.loadouts and m.loadout != want:
            return Action("set_loadout", f"loadout {want} for {g}", want)
        return Action("press_play", f"lobby, goal {g}")
    if scr == "modes_menu":
        return {"raid": Action("open_raid", "modes menu: RAID card (teleports to the raid world)"),
                "speed": Action("open_speed", "modes menu: Speed Raid card"),
                "infinity": Action("open_dimension", "modes menu, goal Infinity"),
                "dimension": Action("open_dimension", "modes menu, goal Dimension")}[g]
    if scr == "raid_world":
        if g != "raid":
            return Action("wait", f"raid world, goal {g}: restart after {RAID_WORLD_GRACE} s (no way back)")
        if facts.gf_queued:
            if rt is not None and GF_FALLBACK <= rt < 1800:
                return Action("cancel_queue", "still queued 2 min after the raid opened: create it instead")
            return Action("wait", "Group Finder: queued, waiting for the raid")
        return Action("press_play", "raid world: PLAY opens the raid list")
    if scr == "raid_window":
        if g != "raid":
            return Action("close_window", f"raid window, goal {g}")
        if facts.start_shown:
            return Action("start", "PARTY (raid): Start")   # PARTY is labelled as the window it opens from
        key = raid_hour(now, facts.raid_opens_at) if facts.raid_opens_at is not None else None
        gf_late = rt is not None and GF_FALLBACK <= rt < 1800
        if s.raid_entry == "gf" and m.create_hour != key and not gf_late:
            return Action("queue_gf", "Group Finder tab: queue for the raid")
        if rt is not None and (rt >= CREATE_FROM or rt < RAID_WINDOW):
            return Action("create_raid", "Create tab: create the raid (open to others)")
        return Action("wait", "at the Create screen until the raid opens")
    if scr == "party":
        return Action("start", "PARTY: Start")
    if scr == "speed_raid_window":
        if g != "speed":
            return Action("close_window", f"Speed Raid window, goal {g}")
        return Action("speed_start", "Speed Raid: Start, lowest free character, Start")
    if scr == "dimension_window":
        if g in ("infinity", "dimension") and facts.start_shown:
            return Action("start", "PARTY: Start")          # PARTY is labelled as the window it opens from
        if g == "infinity":
            return Action("pick_infinite", "Dimension window: Infinite Mode, then Create")
        if g == "dimension":
            return Action("unhandled", "Dimension window: choosing a Dimension is not in a slice yet")
        return Action("close_window", f"Dimension window, goal {g}")
    return Action("unhandled", f"screen {scr} not in this slice")


class Policy:
    """decide() plus the memory it needs: the entered mode, the run's start, the raid hour done, Speed Raid's time."""

    def __init__(self, settings: Settings | None = None, own_mode: bool = True):
        self.s = settings or Settings()
        self.m = Memory()
        self.own_mode = own_mode   # False in shadow mode: mode and loadout are set from what the old bot actually did

    @property
    def mode(self) -> str | None:
        return self.m.mode

    @mode.setter
    def mode(self, v: str | None) -> None:
        self.m.mode = v

    def _observe(self, now: float, f: Facts) -> None:
        m = self.m
        if f.screen != m.screen or m.screen_since is None:
            m.screen, m.screen_since = f.screen, now
            if f.screen is not None:
                m.esc_done = False
        if f.screen == "results_raid" and m.mode == "raid" and f.raid_opens_at is not None:
            m.raid_end_seen = raid_hour(now, f.raid_opens_at)
        if f.screen == "fight" and m.fight_since is None:
            m.fight_since = now
            if m.mode == "raid" and f.raid_opens_at is not None:
                m.raid_done = raid_hour(now, f.raid_opens_at)
        elif f.screen in RUN_OVER and m.fight_since is not None:
            if m.mode == "speed":
                m.speed_due = m.fight_since + SPEED_EVERY + 15
            m.fight_since = None
        if self.own_mode and f.screen in ("lobby", "modes_menu"):
            m.mode = None

    def step(self, now: float, facts: Facts) -> Action:
        self._observe(now, facts)
        a = decide(now, facts, self.s, self.m)
        m = self.m
        if a.name == "cancel_queue" and facts.raid_opens_at is not None:
            m.create_hour = raid_hour(now, facts.raid_opens_at)
        if a.name == "press_esc":
            m.esc_done = True
        if a.name == "restart_roblox":
            m.last_restart = now
            self._raid_retry(now, facts)
        if self.own_mode:
            if a.name == "pick_infinite":
                m.mode = "infinity"
            elif a.name in ("queue_gf", "create_raid"):
                m.mode = "raid"
            elif a.name == "speed_start":
                m.mode = "speed"
                m.speed_due = now + SPEED_EVERY   # in case the end is not seen; exact time once it ends
            elif a.name == "set_loadout":
                m.loadout = a.arg
            elif a.name == "restart_roblox":
                m.mode, m.fight_since, m.roblox_started = None, None, now
        return a

    def _raid_retry(self, now: float, f: Facts) -> None:
        """A raid fight cut off in its first 10 min gave no rewards: allow one more try this raid hour (never after
        its results were seen: the game allows one raid an hour, and a raid can end before minute 10)."""
        m = self.m
        if m.mode != "raid" or m.fight_since is None or f.raid_opens_at is None:
            return
        key, rt = raid_hour(now, f.raid_opens_at), raid_time(now, f.raid_opens_at)
        if m.raid_done == key and rt < RAID_WINDOW and m.raid_end_seen != key and m.raid_retry != key:
            m.raid_done, m.raid_retry = None, key
