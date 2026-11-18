"""Measure the screen filter (adbot2.smoothing) on the newest replay (tools/replay.py).

    python tools/smoothing_eval.py                      try a few half-lives, report each
    python tools/smoothing_eval.py --batch 60           also write a labelling batch from the replay (see below)

For labelled val/test frames: raw accuracy (one frame) vs filtered accuracy, how often the filter says "unknown",
and how many screen changes per hour each gives on the capture timeline (flicker). Frames more than 30 s apart start
a new run of the filter, as the live bot would after a pause.

--batch N writes <ADBOT2_HOME>/data/labels/batches/replay_<time>.json: unlabelled frames where the filter says
"unknown" or where it disagrees with the old bot's hint, spread out in time (>= 30 s apart), rare screens first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.dataset import LabelStore, load_manifest
from adbot2.smoothing import ScreenFilter

GAP_S = 30.0


def run_filter(times: np.ndarray, probs: np.ndarray, classes: list[str], half_life: float, min_belief: float):
    out, beliefs = [], []
    f = ScreenFilter(classes, half_life_s=half_life, min_belief=min_belief)
    prev = None
    for t, p in zip(times, probs):
        if prev is not None and t - prev > GAP_S:
            f.reset()
        prev = t
        s, b = f.update(p.astype(np.float64), t)
        out.append(s)
        beliefs.append(b)
    return out, np.array(beliefs)


def changes_per_hour(seq: list[str | None], times: np.ndarray) -> float:
    known = [(s, t) for s, t in zip(seq, times) if s is not None]
    n = sum(1 for (a, _), (b, _) in zip(known, known[1:]) if a != b)
    hours = max(1e-6, (times[-1] - times[0]) / 3600)
    return n / hours


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--min-belief", type=float, default=0.7)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import split_map

    rdir = paths.home() / "reports" / "replay"
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8"))
    with np.load(rdir / latest["file"]) as z:
        ids, tstr, probs = z["ids"].tolist(), z["time"].tolist(), z["screen_probs"].astype(np.float32)
    classes = latest["classes"]
    man = load_manifest(paths.manifest_path())
    keep = [k for k, i in enumerate(ids) if i in man and man[i].root == "captures"]
    ids = [ids[k] for k in keep]
    probs = probs[keep]
    times = np.array([datetime.fromisoformat(tstr[k]).timestamp() for k in keep])
    labels = LabelStore(paths.screen_labels_path()).all()
    split = split_map([r for r in man.values() if r.full_frame])
    held = [k for k, i in enumerate(ids) if split.get(i) in ("val", "test") and labels.get(i) in classes]
    raw = [classes[int(p.argmax())] for p in probs]
    print(f"{len(ids)} capture frames replayed; {len(held)} labelled val/test frames among them")
    raw_acc = np.mean([raw[k] == labels[ids[k]] for k in held]) if held else float("nan")
    print(f"raw (one frame):  accuracy {raw_acc:.3f}, {changes_per_hour(raw, times):.0f} screen changes / hour")
    best = None
    for hl in (1.0, 2.0, 3.0, 5.0, 8.0):
        seq, bel = run_filter(times, probs, classes, hl, args.min_belief)
        known = [k for k in held if seq[k] is not None]
        acc = np.mean([seq[k] == labels[ids[k]] for k in known]) if known else float("nan")
        unk = 1 - len(known) / max(1, len(held))
        wrong = sum(seq[k] != labels[ids[k]] for k in known)
        print(f"half-life {hl:>3} s: accuracy when sure {acc:.3f} ({wrong} wrong), unknown {unk:.1%}, "
              f"{changes_per_hour(seq, times):.0f} changes / hour")
        if best is None or (wrong, unk) < best[0]:
            best = ((wrong, unk), hl, seq)

    if args.batch:
        _, hl, seq = best
        cand = []
        for k, i in enumerate(ids):
            if i in labels:
                continue
            hint = man[i].hint
            if seq[k] is None or (hint and seq[k] != hint):
                cand.append((k, seq[k] or raw[k]))
        rare = Counter(s for _, s in cand)
        cand.sort(key=lambda x: rare[x[1]])
        chosen, used = [], []
        for k, s in cand:
            if all(abs(times[k] - u) >= 30 for u in used):
                chosen.append({"id": ids[k], "why": f"replay: filter {seq[k] or 'unknown'}, model {raw[k]}"})
                used.append(times[k])
            if len(chosen) >= args.batch:
                break
        name = f"replay_{datetime.now():%Y%m%d_%H%M}"
        bdir = paths.data_dir() / "labels" / "batches"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{name}.json").write_text(json.dumps({"name": name, "items": chosen}, indent=1), encoding="utf-8")
        print(f"batch {name}: {len(chosen)} frames (from {len(cand)} unknown / disagreeing), half-life {hl} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
