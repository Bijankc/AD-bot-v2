"""Check the AUTO reader on the replay timeline against the old bot's log, no labels needed.

    python tools/auto_check.py [--bot-dir ..\\AD-bot] [--sheet]

- During Auto fights (frames hinted by raid_auto / speed_auto / inf_auto / not_dying, more than 10 s from any Auto
  press): frames with P(off) >= 0.98 would make the bot press Auto. Each is a candidate false OFF; --sheet saves
  their AUTO crops to <ADBOT2_HOME>/reports/auto_check_<time>.png to look at (some are real: Auto does go off).
- Just before the old bot pressed Auto (<= 2.5 s): how many of those frames v2 reads as OFF.
Uses the newest AUTO model's predictions for every cached crop (preds.npz); counts are given for held-out (val/test)
frames separately, since train frames were seen in training.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.crops import HEADS, cut
from adbot2.dataset import load_game_area, load_manifest
from adbot2.oldlog import read_all_lines
import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import split_map  # noqa: E402
from adbot2.perception import AUTO_OFF_ACT, newest

AUTO_RULES = ("raid_auto", "speed_auto", "inf_auto", "not_dying")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args(argv)
    run = newest(paths.home() / "models" / "crops" / "auto", "preds.npz")
    with np.load(run / "preds.npz") as z:
        pid, pp = z["ids"].tolist(), z["probs"].astype(np.float32)
    off = HEADS["auto"].classes.index("off")
    man = load_manifest(paths.manifest_path())
    order = sorted((k for k, i in enumerate(pid) if i in man and man[i].time), key=lambda k: man[pid[k]].time)
    ids = [pid[k] for k in order]
    tstr = [man[i].time for i in ids]
    auto = pp[order]
    split = split_map([r for r in man.values() if r.full_frame])
    print(f"model {run.name}; frames by split: "
          f"{ {s: sum(split.get(i) == s for i in ids) for s in ('train', 'val', 'test')} }")
    press = sorted(ln.time for ln in read_all_lines(args.bot_dir) if "Auto pressed (try" in ln.text)

    def near_press(t: datetime, before: float, after: float) -> bool:
        k = bisect.bisect_left(press, t)
        later = k < len(press) and (press[k] - t).total_seconds() <= before
        earlier = k > 0 and (t - press[k - 1]).total_seconds() <= after
        return later or earlier

    stats = {g: [0, 0, 0, 0] for g in ("held-out", "train")}   # fight, false off, before press, seen off
    false_off = []
    for i, ts, p in zip(ids, tstr, auto):
        rec = man[i]
        g = "train" if split.get(i) == "train" else "held-out"
        t = datetime.fromisoformat(ts)
        if near_press(t, 2.5, 0.0):
            stats[g][2] += 1
            stats[g][3] += p[off] >= AUTO_OFF_ACT
        elif rec.rule in AUTO_RULES and not near_press(t, 10.0, 10.0):
            stats[g][0] += 1
            if p[off] >= AUTO_OFF_ACT:
                stats[g][1] += 1
                false_off.append((float(p[off]), i, t))
    for g, (fight, fo, bp, so) in stats.items():
        print(f"{g:8}: Auto fights {fight} frames, {fo} read as OFF at P >= {AUTO_OFF_ACT}; "
              f"just before the old bot pressed Auto: {so} of {bp} read as OFF")
    for p, i, t in sorted(false_off, reverse=True)[:10]:
        print(f"  {t:%d.%m %H:%M:%S} P(off) {p:.3f}  {i}")
    if args.sheet and false_off:
        tiles = []
        for p, i, t in sorted(false_off, reverse=True)[:40]:
            c = cut(load_game_area(man[i], paths.home()).image, HEADS["auto"]).copy()
            cv2.putText(c, f"{p:.2f}", (2, 115), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
            tiles.append(cv2.copyMakeBorder(c, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 0, 255)))
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        out = paths.home() / "reports" / f"auto_check_{datetime.now():%Y%m%d_%H%M}.png"
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]))
        print("sheet:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
