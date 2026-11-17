"""Evaluate a screen predictor on held-out sessions and the regression set; write a report with the worst examples.

    python tools/evaluate.py --predictor hint     the old bot's own belief (its log), as a baseline
    python tools/evaluate.py --predictor knn      nearest neighbours on the embeddings, trained on the train split

Splits are frozen in <ADBOT2_HOME>/data/splits.json (new sessions are added, none ever move). Reports go to
<ADBOT2_HOME>/reports/<time>_<predictor>/ (report.md, confusion.csv, worst/*.jpg); never into the repo.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.dataset import LabelStore, load_game_area, load_manifest, read_jsonl
from adbot2.evaluation import (REGRESSION_CASES, assign_splits, guarded, layout_key, load_frozen, regression_set,
                               save_frozen, score, sessions)
from adbot2.geometry import model_view
from adbot2.screens import Screen
from adbot2.selection import knn_probs

CLASSES = [s.value for s in Screen if s is not Screen.UNSURE]


def split_map(recs) -> dict[str, str]:
    """id -> train | val | test | guard, updating the frozen session file."""
    sess = sessions(recs)
    path = paths.data_dir() / "splits.json"
    splits = assign_splits(sess, load_frozen(path))
    save_frozen(path, splits)
    out = {fid: splits[s] for fid, s in sess.items()}
    for fid in guarded(recs, out):
        out[fid] = "guard"
    return out


def predict_hint(recs, labels, split) -> dict[str, tuple[str | None, float]]:
    return {r.id: (r.hint, 1.0 if r.hint else 0.0) for r in recs}


def predict_knn(recs, labels, split) -> dict[str, tuple[str | None, float]]:
    from embed_frames import load_embeddings

    ids, vecs = load_embeddings()
    pos = {fid: i for i, fid in enumerate(ids)}
    train = [fid for fid, lab in labels.items() if split.get(fid) == "train" and fid in pos and lab != "unsure"]
    if len(train) < 5:
        raise SystemExit(f"knn needs labelled train frames (have {len(train)})")
    xl = vecs[[pos[f] for f in train]].astype(np.float32)
    yl = [labels[f] for f in train]
    classes = sorted(set(yl))
    todo = [r.id for r in recs if r.id in pos]
    probs = knn_probs(vecs[[pos[f] for f in todo]].astype(np.float32), xl, yl, classes)
    return {fid: (classes[int(p.argmax())], float(p.max())) for fid, p in zip(todo, probs)}


PREDICTORS = {"hint": predict_hint, "knn": predict_knn}


def save_example(rec, path: Path, text: str) -> None:
    v = cv2.resize(model_view(load_game_area(rec, paths.home())), (960, 509), interpolation=cv2.INTER_AREA)
    v = cv2.copyMakeBorder(v, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(v, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictor", choices=sorted(PREDICTORS), default="hint")
    ap.add_argument("--worst", type=int, default=40, help="wrong examples saved as images")
    args = ap.parse_args(argv)

    man = load_manifest(paths.manifest_path())
    recs = [r for r in man.values() if r.full_frame]
    labels = {k: v for k, v in LabelStore(paths.screen_labels_path()).all().items() if k in man}
    split = split_map(recs)
    preds = PREDICTORS[args.predictor]([man[i] for i in labels], labels, split)

    out = paths.home() / "reports" / f"{datetime.now():%Y%m%d_%H%M}_{args.predictor}"
    lines = [f"# Screen reading: {args.predictor}  ({datetime.now():%d %b %H:%M})", ""]
    wrong = []
    for sp in ("val", "test"):
        for lay in ("100", "150", "shot"):
            ids = [i for i, lab in labels.items() if split.get(i) == sp and layout_key(man[i]) == lay and lab != "unsure"]
            if not ids:
                continue
            s = score([labels[i] for i in ids], [preds.get(i, (None, 0))[0] for i in ids], CLASSES)
            lines += [f"## {sp}, layout {lay}: {s.accuracy:.1%} of {s.n}", ""]
            lines += [f"- {c}: {acc:.1%} of {n}" for c, (acc, n) in s.per_class().items() if n]
            lines += ["", "```", s.table(), "```", ""]
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"confusion_{sp}_{lay}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["true\\pred"] + s.classes + ["none"])
                for i, c in enumerate(s.classes):
                    w.writerow([c] + list(s.confusion[i]) + [s.unanswered[i]])
            wrong += [(preds.get(i, (None, 0))[1], i, sp) for i in ids if preds.get(i, (None, 0))[0] != labels[i]]

    cands = read_jsonl(paths.data_dir() / "labels" / "regression_candidates.jsonl")
    reg = regression_set(cands, labels)
    by_case = defaultdict(list)
    for e in reg:
        by_case[e["case"]].append(e)
    lines += ["## Regression set", ""]
    failures = 0
    for case, desc in REGRESSION_CASES.items():
        es = by_case.get(case, [])
        bad = [e for e in es if preds.get(e["id"], (None, 0))[0] != e["screen"]]
        failures += len(bad)
        state = "no frames yet" if not es else ("ok" if not bad else f"FAIL {len(bad)}")
        lines.append(f"- {case} ({len(es)}): {state}  - {desc}")
        for e in bad:
            wrong.append((9.0, e["id"], f"regression_{case}"))
    lines.append("")

    wrong.sort(key=lambda t: -t[0])  # most confident mistakes (and regression failures) first
    for n, (conf, fid, where) in enumerate(wrong[:args.worst]):
        p = preds.get(fid, (None, 0))[0]
        save_example(man[fid], out / "worst" / f"{n:02d}_{where}_{labels[fid]}_as_{p}.jpg",
                     f"{where}: true {labels[fid]}, read {p} ({conf if conf < 9 else 1:.2f})")
    lines.append(f"{len(wrong)} wrong, {min(len(wrong), args.worst)} saved in worst/; regression failures: {failures}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
