"""Pick the next labelling batch (diverse, balanced, plus what the labels disagree on) and report label counts.

    python tools/pick_batch.py              new batch of 120 -> <ADBOT2_HOME>/data/labels/batches/<name>.json
    python tools/pick_batch.py --report     only the counts: labels per screen, and how often each log rule agrees

Run tools/import_captures.py and tools/embed_frames.py first. Then label with tools/label_screens.py (newest batch).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from adbot2 import paths
from adbot2.dataset import LabelStore, load_manifest
from adbot2.screens import Screen
from adbot2.selection import Candidate, pick
from embed_frames import load_embeddings


def batches_dir():
    return paths.data_dir() / "labels" / "batches"


def report(recs: dict, labels: dict[str, str]) -> str:
    c = Counter(labels.values())
    out = [f"labels: {sum(c.values())} frames"]
    out += [f"  {s.value:18} {c.get(s.value, 0):5}" for s in Screen]
    agree: dict[str, Counter] = defaultdict(Counter)
    for fid, screen in labels.items():
        r = recs.get(fid)
        if r and r.hint and screen != "unsure":
            agree[r.rule or f"reason:{r.reason}"]["ok" if screen == r.hint else screen] += 1
    if agree:
        out.append("hint agreement per rule (labelled frames):")
        for rule, cnt in sorted(agree.items()):
            n = sum(cnt.values())
            wrong = ", ".join(f"{k} {v}" for k, v in cnt.most_common() if k != "ok")
            out.append(f"  {rule:20} {cnt['ok']}/{n} agree" + (f"; labelled {wrong}" if wrong else ""))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    recs = load_manifest(paths.manifest_path())
    labels = LabelStore(paths.screen_labels_path()).all()
    if args.report:
        print(report(recs, labels))
        return 0
    ids, vecs = load_embeddings()
    pos = {fid: i for i, fid in enumerate(ids)}
    cands, rows = [], []
    for r in recs.values():
        if r.full_frame and r.id in pos:
            t = datetime.fromisoformat(r.time) if r.time else None
            group = r.file.split("/")[0] if r.root == "captures" else r.source
            cands.append(Candidate(r.id, t, group, r.hint))
            rows.append(pos[r.id])
    missing = sum(1 for r in recs.values() if r.full_frame and r.id not in pos)
    if missing:
        print(f"note: {missing} frames have no embedding yet (run tools/embed_frames.py)")
    x = vecs[rows].astype(np.float32)
    chosen = pick(cands, x, labels, n=args.n, clusters=args.clusters)
    name = f"{datetime.now():%Y%m%d_%H%M}"
    batches_dir().mkdir(parents=True, exist_ok=True)
    (batches_dir() / f"{name}.json").write_text(json.dumps(
        {"name": name, "made": datetime.now().isoformat(timespec="seconds"),
         "items": [{"id": i, "why": w} for i, w in chosen]}, indent=1), encoding="utf-8")
    by_hint = Counter(recs[i].hint or "-" for i, _ in chosen)
    kinds = Counter(w.split(" ")[0] for _, w in chosen)
    print(f"batch {name}: {len(chosen)} frames ({', '.join(f'{k} {v}' for k, v in kinds.items())})")
    print("  by hint: " + ", ".join(f"{k} {v}" for k, v in by_hint.most_common()))
    print(report(recs, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
