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
from datetime import datetime
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.oldlog import read_all_lines
from adbot2.perception import Observation
from adbot2.policy import Policy, goal
from adbot2.shadowing import OldBotState
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
GOAL_LINE = re.compile(r"^goal: (.*)")


def at_time(tl, times, t, default=None):
    j = bisect.bisect_right(times, t)
    return tl[j - 1][1] if j else default


GAP_S = 30.0
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
    goals = [(ln.time.timestamp(), m.group(1)) for ln in lines if (m := GOAL_LINE.search(ln.text))]
    goal_t = [t for t, _ in goals]
    old_bot, li = OldBotState(), 0
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
                        start_button=bool(data["start"][k]) if "start" in data else None,
                        menu_clock=None if data["clock"][k] < 0 else divmod(int(data["clock"][k]), 60))
        f = tr.update(o, t)
        while li < len(lines) and lines[li].time.timestamp() <= t:
            old_bot.feed(lines[li])
            li += 1
        old_bot.apply(pol)
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
