"""Evaluation: a frozen train/val/test split by session, metrics, and the named regression set.

Neighbouring frames are near-identical, so a random split would fake the accuracy. A session is one "farm day" (noon to
noon, so a night stays whole) of one screen layout, and each whole session goes to one split. Per layout, sessions take
turns in time order: train, test, val, train, train, and again. Assignments are frozen in a file and never move, so both
layouts get held-out days. Frames of val/test within GUARD of a train frame of the same layout are left out of the
scores (a session cut at noon would otherwise put near-identical frames on both sides).
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DAY_START = timedelta(hours=12)       # a farm day runs noon to noon
GUARD = timedelta(minutes=3)
PATTERN = ["train", "test", "val", "train", "train"]


def layout_key(rec) -> str:
    """100 or 150 (screen grabs from the old bot carry no DPI: 'shot')."""
    if rec.dpi:
        return "100" if rec.dpi == 96 else "150"
    return "shot"


def sessions(recs) -> dict[str, str]:
    """id -> session name "<farm day>_<layout>", e.g. "20261114_150". Frames without a time: "notime_shot"."""
    out = {}
    for r in recs:
        if r.time:
            out[r.id] = f"{datetime.fromisoformat(r.time) - DAY_START:%Y%m%d}_{layout_key(r)}"
        else:
            out[r.id] = f"notime_{layout_key(r)}"
    return out


def assign_splits(session_of: dict[str, str], frozen: dict[str, str]) -> dict[str, str]:
    """session -> split. Keeps every frozen assignment; a new session gets PATTERN[its index within its layout]."""
    result = dict(frozen)
    by_layout: dict[str, list[str]] = defaultdict(list)
    for s in sorted(set(session_of.values()) | set(frozen)):
        by_layout[s.rsplit("_", 1)[-1]].append(s)
    for names in by_layout.values():
        for k, s in enumerate(names):
            result.setdefault(s, PATTERN[k % len(PATTERN)])
    return result


def guarded(recs, split_of: dict[str, str]) -> set[str]:
    """ids of val/test frames within GUARD of a train frame of the same layout (left out of the scores)."""
    train_times: dict[str, list[datetime]] = defaultdict(list)
    for r in recs:
        if r.time and split_of.get(r.id) == "train":
            train_times[layout_key(r)].append(datetime.fromisoformat(r.time))
    for v in train_times.values():
        v.sort()
    out = set()
    for r in recs:
        if not r.time or split_of.get(r.id) not in ("val", "test"):
            continue
        t, ts = datetime.fromisoformat(r.time), train_times[layout_key(r)]
        k = bisect.bisect_left(ts, t - GUARD)
        if k < len(ts) and ts[k] <= t + GUARD:
            out.add(r.id)
    return out


def load_frozen(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_frozen(path: Path, splits: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(splits.items())), indent=1), encoding="utf-8")


# ---------------------------------------------------------------- metrics

@dataclass
class Scores:
    classes: list[str]
    confusion: np.ndarray        # rows = true, cols = predicted
    unanswered: np.ndarray       # per true class: frames with no prediction (None / unknown class)
    n: int

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.confusion) / self.n) if self.n else 0.0

    def per_class(self) -> dict[str, tuple[float, int]]:
        """class -> (recall, support)."""
        out = {}
        for i, c in enumerate(self.classes):
            sup = int(self.confusion[i].sum() + self.unanswered[i])
            out[c] = (float(self.confusion[i, i] / sup) if sup else float("nan"), sup)
        return out

    def table(self) -> str:
        w = max(len(c) for c in self.classes)
        head = " " * (w + 2) + " ".join(f"{c[:5]:>5}" for c in self.classes) + "  none"
        rows = [head] + [f"{c:>{w}}  " + " ".join(f"{v:5d}" for v in self.confusion[i]) + f" {self.unanswered[i]:5d}"
                         for i, c in enumerate(self.classes)]
        return "\n".join(rows)


def score(y_true: list[str], y_pred: list[str | None], classes: list[str] | None = None) -> Scores:
    """None as a prediction counts as wrong (shown as the column 'none')."""
    classes = list(classes or sorted(set(y_true) | {p for p in y_pred if p}))
    cols = classes + ["none"]
    m = np.zeros((len(classes), len(cols)), int)
    for t, p in zip(y_true, y_pred):
        if t not in classes:
            continue
        m[classes.index(t), cols.index(p if p in classes else "none")] += 1
    return Scores(classes, m[:, :len(classes)], m[:, -1], int(m.sum()))


# ---------------------------------------------------------------- regression set

# Named failures from the old bot's history (its CLAUDE.md). Each case lists what a frame in it must be read as.
REGRESSION_CASES: dict[str, str] = {
    "white_flash": "bright effects in a fight: must stay a fight, never lobby / LEAVE",
    "bright_floor": "bright floor in a fight: must stay a fight",
    "auto_font": "AUTO in the old and new fonts, at 100% and 150%: a fight with AUTO",
    "auto_off_diamond": "AUTO OFF with a dark and with a red diamond",
    "dimmed_menu": "a menu dimmed by a window over it",
    "gf_banner": "Group Finder queue banner, both forms (queued / Group found)",
    "results_raid": "RAID RESULTS (solo)",
    "results_raid_gf": "Group Finder raid results (ranking left, small cards)",
    "results_infinity": "Infinity results with RETRY / LEAVE / AUTO RETRY",
    "results_dimension": "Dimension results (rank, rewards, RETRY / LEAVE)",
    "results_only_leave": "Dimension results with only LEAVE",
    "revive": "death screen with REVIVE (never pressed)",
    "raid_clock": "raid clock digits (modes menu card and raid-world banner)",
}


def regression_set(candidates: list[dict], labels: dict[str, str]) -> list[dict]:
    """Confirmed regression frames: candidates {id, case} the human has labelled (not 'unsure')."""
    out = []
    for c in candidates:
        lab = labels.get(c["id"])
        if lab and lab != "unsure":
            out.append({"id": c["id"], "case": c["case"], "screen": lab})
    return out
