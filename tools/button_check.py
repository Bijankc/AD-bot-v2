"""Check the colour-rule button readers (adbot2.buttons) on stored frames, with no labels.

    python tools/button_check.py [--bot-dir ..\\AD-bot] [--others 600]

Positives: frames up to 1.5 s before the old bot logged "pressing Start" (the PARTY window is on screen then).
Negatives: every other frame the newest replay calls a window (dimension, raid, Speed Raid, Extreme, party), plus a
random sample of all other frames. Prints the orange / white shares per set and lists every negative read as
shown (and near misses), so the thresholds in adbot2.buttons can be set with eyes on the numbers. Frames are
decoded 64 at a time at below-normal priority and the job stops under the RAM floor (adbot2.resources).
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.buttons import read_start
from adbot2.capture import lower_priority
from adbot2.dataset import load_game_area, load_manifest
from adbot2.geometry import UnknownLayout
from adbot2.oldlog import read_all_lines
from adbot2.resources import ram_message, ram_ok

WINDOWS = {"dimension_window", "raid_window", "party", "speed_raid_window", "extreme_window"}
BEFORE_START_S = 1.5


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--others", type=int, default=600, help="random non-window frames")
    args = ap.parse_args(argv)
    if not ram_ok():
        print(ram_message())
        return 2
    lower_priority()
    man = sorted((r for r in load_manifest(paths.manifest_path()).values() if r.full_frame and r.time),
                 key=lambda r: r.time)
    ts = [datetime.fromisoformat(r.time).timestamp() for r in man]
    pos: set[str] = set()
    for ln in read_all_lines(args.bot_dir):
        if re.search(r"^pressing Start", ln.text):
            t = ln.time.timestamp()
            lo, hi = bisect.bisect_left(ts, t - BEFORE_START_S), bisect.bisect_right(ts, t)
            pos |= {man[j].id for j in range(lo, hi)}
    rdir = paths.home() / "reports" / "replay"
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8"))
    with np.load(rdir / latest["file"]) as z:
        pred = dict(zip(z["ids"].tolist(), (latest["classes"][k] for k in z["screen_probs"].argmax(1))))
    rest = [r for r in man if r.id not in pos]
    random.seed(0)
    others = [r for r in rest if pred.get(r.id) not in WINDOWS]
    sets = {"start": [r for r in man if r.id in pos],
            "window": [r for r in rest if pred.get(r.id) in WINDOWS],
            "other": random.sample(others, min(args.others, len(others)))}

    def read(r):
        try:
            return read_start(load_game_area(r, paths.home()).image)
        except (UnknownLayout, OSError, ValueError):
            return None

    results = {}
    with ThreadPoolExecutor(4) as ex:
        for name, recs in sets.items():
            out = []
            for i in range(0, len(recs), 64):
                if not ram_ok():
                    print(ram_message())
                    return 2
                out += [(r, b) for r, b in zip(recs[i:i + 64], ex.map(read, recs[i:i + 64])) if b is not None]
            results[name] = out
    for name, out in results.items():
        if not out:
            print(f"{name:7} no frames")
            continue
        o = np.array([b.orange for _, b in out])
        w = np.array([b.white for _, b in out])
        print(f"{name:7} n={len(out):5}  shown {sum(b.shown for _, b in out):5}   orange min/median/max "
              f"{o.min():.2f}/{np.median(o):.2f}/{o.max():.2f}   white {w.min():.2f}/{np.median(w):.2f}/{w.max():.2f}")
    print("start frames read as not shown:")
    for r, b in results["start"]:
        if not b.shown:
            print(f"   {r.time} orange {b.orange:.2f} white {b.white:.2f} {r.id}")
    print("other frames read as shown or close (orange >= 0.4):")
    for name in ("window", "other"):
        for r, b in results[name]:
            if b.shown or b.orange >= 0.4:
                print(f"   {name:6} {r.time} {pred.get(r.id)} orange {b.orange:.2f} white {b.white:.2f} "
                      f"{'SHOWN' if b.shown else ''} {r.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
