r"""Register the frame logger's captures in the frame store, with weak screen hints from the old bot's log.

    python tools/import_captures.py --bot-dir "..\AD-bot"

Frames stay where the logger wrote them (no copies); the manifest points at captures/<day>/<file>. Only frames older
than --settle-min are taken, so the log already shows what came after them. Safe to run again (new frames only).
With --rehint, hints of every stored frame are recomputed from the current rules (after a rule or class change).
Read-only on the old bot's folder; each log file is opened and closed at once.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from adbot2 import paths
from adbot2.dataset import FrameRecord, append_jsonl, load_manifest, rewrite_manifest
from adbot2.geometry import layout_of
from adbot2.oldlog import read_all_lines
from adbot2.oldlog import screen_hint
from adbot2.weaklabels import HintIndex, context_at, intervals


def capture_id(day: str, name: str) -> str:
    return f"cap{day}_{Path(name).stem}"


def index_rows(day_dir: Path) -> list[dict]:
    rows = []
    for idx in sorted(day_dir.glob("index*.csv")):
        with open(idx, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def new_records(captures: Path, known: set[str], lines, before: datetime) -> list[FrameRecord]:
    hints = HintIndex(intervals(lines))
    out = []
    for day_dir in sorted(p for p in captures.iterdir() if p.is_dir() and p.name.isdigit()):
        for row in index_rows(day_dir):
            fid = capture_id(day_dir.name, row["file"])
            if fid in known:
                continue
            t = datetime.fromisoformat(row["time"])
            if t > before or not (day_dir / row["file"]).is_file():
                continue
            hint, rule = hints.at(t)
            rect = [int(row[k]) for k in ("left", "top", "width", "height")]
            usable = layout_of(rect[2], rect[3], int(row["dpi"])) is not None  # not mid-resize etc.
            out.append(FrameRecord(fid, f"{day_dir.name}/{row['file']}", "capture", row["file"], rect[2], rect[3],
                                   usable, t.isoformat(timespec="milliseconds"), row.get("kind"), hint,
                                   context_at(lines, t), "captures", rect, int(row["dpi"]), rule))
            known.add(fid)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", required=True, type=Path)
    ap.add_argument("--settle-min", type=float, default=15.0)
    ap.add_argument("--rehint", action="store_true", help="recompute the hints of all stored frames first")
    args = ap.parse_args(argv)
    lines = read_all_lines(args.bot_dir)
    if args.rehint:
        man = load_manifest(paths.manifest_path())
        hints = HintIndex(intervals(lines))
        changed = Counter()
        for r in man.values():
            if r.root == "captures":
                new = hints.at(datetime.fromisoformat(r.time))
            else:
                new = (screen_hint(r.reason), None)
            if (r.hint, r.rule) != new:
                changed[f"{r.hint} -> {new[0]}"] += 1
                r.hint, r.rule = new
        rewrite_manifest(paths.manifest_path(), list(man.values()))
        print("rehinted: " + (", ".join(f"{k} {v}" for k, v in changed.most_common()) or "no change"))
    known = set(load_manifest(paths.manifest_path()))
    recs = new_records(paths.home() / "captures", known, lines, datetime.now() - timedelta(minutes=args.settle_min))
    append_jsonl(paths.manifest_path(), [asdict(r) for r in recs])
    c = Counter(r.hint or "-" for r in recs)
    print(f"added {len(recs)} capture frames; hints: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
