"""Shadow mode on recorded frames: run tracker + policy over the replay timeline and compare with the old bot.

    python tools/shadow.py [--bot-dir ..\\AD-bot] [--opens 2700]

For every action the old bot logged (OLD_ACTIONS: PLAY, loadouts, Dimension, Infinity, Start, Auto, the raid world,
Group Finder, Speed Raid, Roblox restarts other than preventive ones), was v2's decision the same on a frame from a few
seconds before (more for lines logged after the click) to 0.5 s after it? And how often v2 wanted an action the old
bot did not take near it ("extra"; some are fine: frames are ~2 s apart and the old bot also waits between clicks).
Mode and loadout come from the old bot's log (v2 does not choose the run in shadow mode); the schedule memory (raid
done this hour, next Speed Raid) is learned from the frames and is lost at a gap. Frames more than 30 s apart restart the tracker. The raid clock comes from the
menu-clock readings in the replay; --opens (second of the hour) is used until one is confirmed, as the old bot keeps
its last raidOfs in its settings. Nothing is sent to the game: this only reads stored frames.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.oldlog import read_all_lines
from adbot2.perception import Observation
from adbot2.policy import Policy, Settings, goal
from adbot2.tracker import Tracker

# old-bot log line -> v2 action, and how long before the line v2's decision may come (lines logged after the
# click: "entering Dimension", "teleporting to the raid world", "Speed Raid: start" come seconds after it)
OLD_ACTIONS = [
    (re.compile(r"^pressing PLAY"), "press_play", 3.0),
    (re.compile(r"^loadout \d: opening Bag"), "set_loadout", 3.0),
    (re.compile(r"^entering Dimension"), "open_dimension", 3.0),
    (re.compile(r"^pressing Create \(infinity\)"), "pick_infinite", 3.0),
    (re.compile(r"^pressing Start"), "start", 3.0),
    (re.compile(r"Auto pressed \(try"), "press_auto", 3.0),
    (re.compile(r"^leaving Infinity \("), "reset", 3.0),
    (re.compile(r"^(Infinity|Speed Raid) results: pressing LEAVE"), "leave", 3.0),
    (re.compile(r"^teleporting to the raid world"), "open_raid", 45.0),
    (re.compile(r"^Group Finder tab"), "queue_gf", 3.0),
    (re.compile(r"^Speed Raid: opening"), "open_speed", 3.0),
    (re.compile(r"^Speed Raid: start \(character"), "speed_start", 15.0),
    (re.compile(r"^restarting Roblox: (?!preventive)"), "restart_roblox", 3.0),
]
LOADOUT_LINE = re.compile(r"^loadout (\d): opening Bag")   # the intent: "loadout N ✓" comes with PLAY
SPEED_AT = re.compile(r"next Speed Raid at (\d\d):(\d\d)")
SPEED_NOW = re.compile(r"\(Speed Raid is available\)")
GOAL_LINE = re.compile(r"^goal: (.*)")
START_LINE = re.compile(r"^start v.*loadouts=(\d), raid=(\d), infinity=(\d), speed=(\d), raidEntry=(\w+)")


def timeline(lines, parse):
    """[(t, value)] for every line where parse(line) is not None, in time order."""
    out = []
    for ln in lines:
        v = parse(ln)
        if v is not None:
            out.append((ln.time.timestamp(), v))
    return out


def speed_due(ln):
    """When the old bot planned its next Speed Raid, from its goal lines."""
    if SPEED_NOW.search(ln.text):
        return ln.time.timestamp()
    if m := SPEED_AT.search(ln.text):
        due = ln.time.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if due < ln.time - timedelta(hours=12):
            due += timedelta(days=1)
        return due.timestamp()
    return None


def settings(ln):
    if m := START_LINE.search(ln.text):
        lo, raid, inf, speed, entry = m.groups()
        # infinity=0 was the Titan farming days (titan is not in the start line)
        return Settings(raid=raid == "1", speed=speed == "1", infinity=inf == "1", dimension=inf == "0",
                        raid_entry=entry, loadouts=lo == "1")
    return None


def at_time(tl, times, t, default=None):
    j = bisect.bisect_right(times, t)
    return tl[j - 1][1] if j else default
GAP_S = 30.0
# what the old bot entered, for the policy's mode memory (v2 does not choose the run in shadow mode)
ENTERED = [
    (re.compile(r"^pressing Create \(infinity\)"), "infinity"),
    (re.compile(r"^Titan: pressing Create"), "titan"),
    (re.compile(r"^Speed Raid: start \(character"), "speed"),
    (re.compile(r"^(pressing Create \(raid\)|Group Finder: raid open)"), "raid"),
    (re.compile(r"^(pressing PLAY|lobby \(PLAY visible\)|restarting Roblox)"), None),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--opens", type=int, default=2700, help="second of the hour the raid opens, until read")
    ap.add_argument("--why", metavar="ACTION", help="print every miss of this old-bot action with v2's view")
    args = ap.parse_args(argv)
    rdir = paths.home() / "reports" / "replay"
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8"))
    with np.load(rdir / latest["file"]) as z:
        data = {k: z[k] for k in z.files}
    classes = latest["classes"]
    times = np.array([datetime.fromisoformat(t).timestamp() for t in data["time"].tolist()])
    order = np.argsort(times)

    lines = read_all_lines(args.bot_dir)
    entered = []
    for ln in lines:
        for rx, mode in ENTERED:
            if rx.search(ln.text):
                entered.append((ln.time.timestamp(), mode))
                break
    entered_t = [t for t, _ in entered]
    loadouts = timeline(lines, lambda ln: int(m.group(1)) if (m := LOADOUT_LINE.search(ln.text)) else None)
    speeds = timeline(lines, speed_due)
    configs = timeline(lines, settings)
    loadout_t, speed_t, config_t = ([t for t, _ in tl] for tl in (loadouts, speeds, configs))

    goals = timeline(lines, lambda ln: m.group(1) if (m := GOAL_LINE.search(ln.text)) else None)
    goal_t = [t for t, _ in goals]
    decisions: list[tuple[float, str]] = []
    views: list[str] = []
    tr, prev = None, None
    for k in order:
        t = float(times[k])
        if tr is None or t - prev > GAP_S:
            tr = Tracker(classes)
            tr.facts.raid_opens_at = args.opens
            pol = Policy(own_mode=False)
        prev = t
        auto = data["auto_probs"][k].astype(np.float32)
        o = Observation(screen_probs=data["screen_probs"][k].astype(np.float32), auto_p_off=float(auto[1]),
                        enemies=bool(data["enemies"][k]),
                        enemy_count=None if data["count"][k] < 0 else int(data["count"][k]),
                        gf_banner=bool(data["gf"][k]),
                        menu_clock=None if data["clock"][k] < 0 else divmod(int(data["clock"][k]), 60))
        f = tr.update(o, t)
        j = bisect.bisect_right(entered_t, t)
        pol.mode = entered[j - 1][1] if j else None
        pol.m.loadout = at_time(loadouts, loadout_t, t)
        pol.m.speed_due = at_time(speeds, speed_t, t)
        pol.s = at_time(configs, config_t, t, Settings())
        a = pol.step(t, f)
        decisions.append((t, a.name))
        if args.why:
            views.append(f"{f.screen} opens={f.raid_opens_at} clock={o.menu_clock} mode={pol.mode} goal={goal(t, f, pol.s, pol.m)} -> {a.name}: {a.why}")
    dt = [t for t, _ in decisions]

    old = []
    for ln in lines:
        for rx, name, before in OLD_ACTIONS:
            if rx.search(ln.text):
                old.append((ln.time.timestamp(), name, before))
                break
    covered = Counter()
    matched = Counter()
    seen_instead: dict[str, Counter] = defaultdict(Counter)
    for t, name, before in old:
        lo, hi = bisect.bisect_left(dt, t - before), bisect.bisect_right(dt, t + 0.5)
        if lo == hi:
            continue  # no frame near this action (Roblox not in front, or before the logger ran)
        covered[name] += 1
        got = [decisions[j][1] for j in range(lo, hi)]
        if name in got:
            matched[name] += 1
        else:
            seen_instead[name].update(got)
            if name == args.why:
                print(f"{datetime.fromtimestamp(t):%d.%m %H:%M:%S} old goal: {at_time(goals, goal_t, t)}")
                for j in range(lo, hi):
                    print(f"    {datetime.fromtimestamp(decisions[j][0]):%H:%M:%S} {views[j]}")
    old_times = defaultdict(list)
    for t, name, _ in old:
        old_times[name].append(t)
    extra = Counter()
    for t, name in decisions:
        if name in ("wait", "unhandled"):
            continue
        ts = old_times.get(name, [])
        j = bisect.bisect_left(ts, t - 10)
        if not (j < len(ts) and ts[j] <= t + 50):
            extra[name] += 1
    hours = (dt[-1] - dt[0]) / 3600 if dt else 0
    print(f"{len(decisions)} frames over {hours:.0f} h; v2 decisions: {dict(Counter(n for _, n in decisions))}")
    print(f"{'old bot action':16} {'with frames':>11} {'v2 same':>8}   v2 decided instead (top)")
    for _, name, _ in OLD_ACTIONS:
        if covered[name]:
            print(f"{name:16} {covered[name]:11} {matched[name]:8}   {seen_instead[name].most_common(3)}")
    print("extra v2 actions (no such old-bot action within 10 s):", dict(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
