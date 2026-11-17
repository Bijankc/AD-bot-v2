"""Propose frames for the named regression cases (adbot2.evaluation.REGRESSION_CASES) and make a labelling batch of them.

    python tools/regression_candidates.py --bot-dir "..\\AD-bot"

Candidates come from the old bot's screenshot reasons, its log (e.g. frames just before "Auto: off") and simple image
measures (very bright fight frames). They become regression frames only once the user labels them on the labelling
page, so a wrong guess costs one key press. Writes <ADBOT2_HOME>/data/labels/regression_candidates.jsonl (append,
unique per id+case) and a batch "regression_<time>". Cases with no candidate are listed at the end.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.dataset import append_jsonl, frame_path, load_manifest, read_jsonl
from adbot2.evaluation import REGRESSION_CASES
from adbot2.geometry import VALID_BOTTOM, VALID_TOP
from adbot2.oldlog import read_all_lines

PER_CASE = 10
REASON_CASES = {
    "auto_blind": "auto_font", "auto_missing": "auto_font", "auto_colour_stuck": "auto_font",
    "gf_banner": "gf_banner", "gf_gone": "gf_banner", "gf_fallback": "gf_banner",
    "raid_end": "results_raid", "raid_rewards": "results_raid", "leave": "results_infinity",
    "xdim_win": "results_dimension", "titan_dead": "results_dimension", "dim_died": "revive",
}


def white_fraction(rec) -> float:
    img = cv2.imdecode(np.fromfile(str(frame_path(rec, paths.home())), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        return 0.0
    h = img.shape[0]
    band = img[int(h * VALID_TOP / 1080):int(h * VALID_BOTTOM / 1080)] if rec.root != "captures" else img
    return float((band > 235).mean())


def spread(ids_times: list[tuple[str, datetime | None]], n: int, gap_s: float = 60, seed: int = 0) -> list[str]:
    """Up to n ids, at least gap_s apart, from all over the list."""
    items = list(ids_times)
    random.Random(seed).shuffle(items)
    out, times = [], []
    for fid, t in items:
        if t and any(abs((t - u).total_seconds()) < gap_s for u in times):
            continue
        out.append(fid)
        if t:
            times.append(t)
        if len(out) >= n:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    recs = [r for r in load_manifest(paths.manifest_path()).values() if r.full_frame]
    t_of = {r.id: datetime.fromisoformat(r.time) if r.time else None for r in recs}
    caps = sorted((r for r in recs if r.root == "captures"), key=lambda r: r.time)
    cap_times = [t_of[r.id] for r in caps]
    lines = read_all_lines(args.bot_dir)
    found: dict[str, list[str]] = defaultdict(list)

    def frames_near(pattern: str, before: float, after: float) -> list[tuple[str, datetime]]:
        rx = re.compile(pattern)
        out = []
        for ln in lines:
            if rx.search(ln.text):
                lo = bisect.bisect_left(cap_times, ln.time - timedelta(seconds=before))
                hi = bisect.bisect_right(cap_times, ln.time + timedelta(seconds=after))
                out += [(caps[k].id, cap_times[k]) for k in range(lo, hi)]
        return out

    # old bot's screenshots, by the reason in their name
    by_case = defaultdict(list)
    for r in recs:
        if r.source == "old_shots" and r.reason in REASON_CASES:
            by_case[REASON_CASES[r.reason]].append((r.id, t_of[r.id]))
    for case, items in by_case.items():
        found[case] += spread(items, PER_CASE // 2, gap_s=1)

    # from the log
    # Auto is off just before the old bot presses it ("Auto: off" lines only exist in older logs)
    found["auto_off_diamond"] += spread(frames_near(r"^Auto: off|Auto pressed \(try", 2.5, 0.0), PER_CASE)
    found["gf_banner"] += spread(frames_near(r"^Group Finder: queued, waiting", -3.0, 60.0), PER_CASE // 2)
    found["gf_banner"] += spread(frames_near(r"^Group Finder: raid open, waiting for the teleport", -1.0, 15.0), 3)
    found["results_infinity"] += spread(frames_near(r"^Infinity results: pressing LEAVE", 3.0, 0.0), PER_CASE // 2)
    found["results_dimension"] += spread(frames_near(r"^Titan results: ", 1.5, 0.0), PER_CASE // 2)
    found["raid_clock"] += spread(frames_near(r"^raid clock: ", 2.0, 0.0), PER_CASE // 2)
    found["raid_clock"] += spread(frames_near(r"^raid still closed, waiting", -2.0, 60.0), PER_CASE // 2)
    found["dimmed_menu"] += spread(frames_near(r"^(modes menu:|Create tab|Group Finder tab|looking for )", 2.5, 0.5),
                                   PER_CASE, gap_s=20)
    rewards = frames_near(r"^results: reading rewards", 0.0, 4.0)
    gf_times = [ln.time for ln in lines if ln.text.startswith("Group Finder: queued")]
    gf, solo = [], []
    for fid, t in rewards:
        k = bisect.bisect_right(gf_times, t)
        (gf if k and t - gf_times[k - 1] < timedelta(minutes=20) else solo).append((fid, t))
    found["results_raid"] += spread(solo, PER_CASE // 2)
    found["results_raid_gf"] += spread(gf, PER_CASE)
    auto_rules = ("raid_auto", "speed_auto", "inf_auto", "not_dying")
    for dpi in (96, 144):
        found["auto_font"] += spread([(r.id, t_of[r.id]) for r in caps if r.rule in auto_rules and r.dpi == dpi], 3, 600)

    # very bright fight frames: a jump from the previous frame = flash, bright neighbours too = floor
    fights = [r for r in caps if r.hint == "fight"]
    wf = {r.id: white_fraction(r) for r in fights}
    flash, floor = [], []
    for prev, cur, nxt in zip(fights, fights[1:], fights[2:]):
        w = wf[cur.id]
        if w < 0.08:
            continue
        if wf[prev.id] < w / 3 and wf[nxt.id] < w / 3:
            flash.append((w, cur.id, t_of[cur.id]))
        elif wf[prev.id] > 0.05 and wf[nxt.id] > 0.05:
            floor.append((w, cur.id, t_of[cur.id]))
    for case, items in (("white_flash", flash), ("bright_floor", floor)):
        items.sort(reverse=True)
        found[case] += spread([(i, t) for _, i, t in items[:60]], PER_CASE, gap_s=120)

    path = paths.data_dir() / "labels" / "regression_candidates.jsonl"
    have = {(c["id"], c["case"]) for c in read_jsonl(path)}
    new = [{"id": i, "case": c} for c, ids in found.items() for i in dict.fromkeys(ids) if (i, c) not in have]
    append_jsonl(path, new)
    name = f"regression_{datetime.now():%Y%m%d_%H%M}"
    items = [{"id": c["id"], "why": f"regression: {c['case']}"} for c in new]
    bdir = paths.data_dir() / "labels" / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"{name}.json").write_text(json.dumps({"name": name, "items": items}, indent=1), encoding="utf-8")
    print(f"batch {name}: {len(items)} new candidates")
    for case in REGRESSION_CASES:
        n = len(dict.fromkeys(found.get(case, [])))
        print(f"  {case:20} {n:3}" + ("   <- none found: needs frames from labelling" if not n else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
