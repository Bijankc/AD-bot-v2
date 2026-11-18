"""Shadow mode on recorded frames: run tracker + policy over the replay timeline and compare with the old bot.

    python tools/shadow.py [--bot-dir ..\\AD-bot] [--opens 2700]

For every action the old bot logged (pressing PLAY, entering Dimension, Create (infinity), Start, Auto pressed,
leaving Infinity, Infinity LEAVE), was v2's decision the same on a frame from 3 s before to 0.5 s after it? And how
often v2 wanted an action the old bot did not take within 10 s ("extra"; some are fine: frames are ~2 s apart and the
old bot also waits between clicks). Frames more than 30 s apart restart the tracker. The raid clock comes from the
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
from adbot2.policy import decide
from adbot2.tracker import Tracker

OLD_ACTIONS = [
    (re.compile(r"^pressing PLAY"), "press_play"),
    (re.compile(r"^entering Dimension"), "open_dimension"),
    (re.compile(r"^pressing Create \(infinity\)"), "pick_infinite"),
    (re.compile(r"^pressing Start"), "start"),
    (re.compile(r"Auto pressed \(try"), "press_auto"),
    (re.compile(r"^leaving Infinity \("), "reset"),
    (re.compile(r"^Infinity results: pressing LEAVE"), "leave"),
]
GAP_S = 30.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--opens", type=int, default=2700, help="second of the hour the raid opens, until read")
    args = ap.parse_args(argv)
    rdir = paths.home() / "reports" / "replay"
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8"))
    with np.load(rdir / latest["file"]) as z:
        data = {k: z[k] for k in z.files}
    classes = latest["classes"]
    times = np.array([datetime.fromisoformat(t).timestamp() for t in data["time"].tolist()])
    order = np.argsort(times)

    decisions: list[tuple[float, str]] = []
    tr, prev = None, None
    for k in order:
        t = float(times[k])
        if tr is None or t - prev > GAP_S:
            tr = Tracker(classes)
            tr.facts.raid_opens_at = args.opens
        prev = t
        auto = data["auto_probs"][k].astype(np.float32)
        o = Observation(screen_probs=data["screen_probs"][k].astype(np.float32), auto_p_off=float(auto[1]),
                        enemies=bool(data["enemies"][k]),
                        enemy_count=None if data["count"][k] < 0 else int(data["count"][k]),
                        menu_clock=None if data["clock"][k] < 0 else divmod(int(data["clock"][k]), 60))
        f = tr.update(o, t)
        decisions.append((t, decide(t, f).name))
    dt = [t for t, _ in decisions]

    old = []
    for ln in read_all_lines(args.bot_dir):
        for rx, name in OLD_ACTIONS:
            if rx.search(ln.text):
                old.append((ln.time.timestamp(), name))
                break
    covered = Counter()
    matched = Counter()
    seen_instead: dict[str, Counter] = defaultdict(Counter)
    for t, name in old:
        lo, hi = bisect.bisect_left(dt, t - 3.0), bisect.bisect_right(dt, t + 0.5)
        if lo == hi:
            continue  # no frame near this action (Roblox not in front, or before the logger ran)
        covered[name] += 1
        got = [decisions[j][1] for j in range(lo, hi)]
        if name in got:
            matched[name] += 1
        else:
            seen_instead[name].update(got)
    old_times = defaultdict(list)
    for t, name in old:
        old_times[name].append(t)
    extra = Counter()
    for t, name in decisions:
        if name in ("wait", "unhandled"):
            continue
        ts = old_times.get(name, [])
        j = bisect.bisect_left(ts, t - 10)
        if not (j < len(ts) and ts[j] <= t + 10):
            extra[name] += 1
    hours = (dt[-1] - dt[0]) / 3600 if dt else 0
    print(f"{len(decisions)} frames over {hours:.0f} h; v2 decisions: {dict(Counter(n for _, n in decisions))}")
    print(f"{'old bot action':16} {'with frames':>11} {'v2 same':>8}   v2 decided instead (top)")
    for _, name in OLD_ACTIONS:
        if covered[name]:
            print(f"{name:16} {covered[name]:11} {matched[name]:8}   {seen_instead[name].most_common(3)}")
    print("extra v2 actions (no such old-bot action within 10 s):", dict(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
