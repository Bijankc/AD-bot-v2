"""Build and check the ENEMIES LEFT digit reader (adbot2.digits) from the cached `enemies` crops.

    python tools/digit_templates.py cluster              group every digit glyph; writes clusters.png to look at
    python tools/digit_templates.py name RUN "0:1,1:0"   cluster -> digit (-1 = skip); fits reader.npz in RUN
    python tools/digit_templates.py check                read every crop the enemies head calls shown; report

Runs live in <ADBOT2_HOME>/models/digits/enemies/<time>/ (never committed). `check` writes readings.npz (id, value,
distance) and check.png (unreadable crops and one-frame blips: a value that differs from both neighbours in time),
and reports how many crops are read, unreadable or blips.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.crops import CropCache
from adbot2.dataset import load_manifest
from adbot2.digits import GLYPH_H, GLYPH_W, DigitReader, glyphs

K = 24


def runs_dir() -> Path:
    return paths.home() / "models" / "digits" / "enemies"


def shown_crops(min_p: float = 0.9) -> tuple[list[str], np.ndarray]:
    """(ids, crops) the newest enemies head model calls shown."""
    preds = sorted(glob.glob(str(paths.home() / "models" / "crops" / "enemies" / "*" / "preds.npz")))
    ids, crops = CropCache(paths.home() / "cache" / "crops").load("enemies")
    if not preds:
        sys.exit("train the enemies head first: tools/train_crops.py --head enemies")
    z = np.load(preds[-1])
    p = dict(zip(z["ids"].tolist(), z["probs"][:, 0]))
    keep = [k for k, i in enumerate(ids) if p.get(i, 0) >= min_p]
    return [ids[k] for k in keep], crops[keep]


def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    tiles = [cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=128) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def cmd_cluster() -> int:
    ids, crops = shown_crops()
    gl = [g for c in crops for g in glyphs(np.asarray(c))]
    x = np.stack([g.reshape(-1) for g in gl]).astype(np.float32)
    cv2.setRNGSeed(0)
    _, lab, _ = cv2.kmeans(x, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 3,
                           cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    run = runs_dir() / f"{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "clusters.npz", x=x.reshape(-1, GLYPH_H, GLYPH_W), lab=lab)
    rows = []
    for c in range(K):
        mem = np.where(lab == c)[0][:12]
        tiles = [cv2.resize((x[i].reshape(GLYPH_H, GLYPH_W) * 255).astype(np.uint8), None, fx=2, fy=2,
                            interpolation=cv2.INTER_NEAREST) for i in mem]
        tiles += [np.zeros((GLYPH_H * 2, GLYPH_W * 2), np.uint8)] * (12 - len(tiles))
        name = np.zeros((GLYPH_H * 2 + 2, 90), np.uint8)
        cv2.putText(name, f"{c}:{int((lab == c).sum())}", (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        rows.append(np.hstack([name, sheet(tiles, 12)]))
    cv2.imwrite(str(run / "clusters.png"), np.vstack(rows))
    print(f"{len(gl)} glyphs from {len(ids)} crops -> {run / 'clusters.png'}")
    return 0


def cmd_name(run: str, mapping: str) -> int:
    folder = runs_dir() / run
    with np.load(folder / "clusters.npz") as z:
        x, lab = z["x"], z["lab"]
    m = {int(a): int(b) for a, b in (p.split(":") for p in mapping.split(","))}
    keep = [i for i, c in enumerate(lab) if m.get(int(c), -1) >= 0]
    reader = DigitReader.fit([x[i] for i in keep], [m[int(lab[i])] for i in keep])
    reader.save(folder / "reader.npz")
    print(f"reader from {len(keep)} glyphs, digits {reader.digits}, max distance {reader.max_dist:.2f}")
    return 0


def cmd_check() -> int:
    readers = sorted(runs_dir().glob("*/reader.npz"))
    if not readers:
        sys.exit("no reader yet: run cluster, then name")
    reader = DigitReader.load(readers[-1])
    ids, crops = shown_crops()
    man = load_manifest(paths.manifest_path())
    reads = [reader.read(np.asarray(c)) for c in crops]
    vals = np.array([-1 if r.value is None else r.value for r in reads])
    dist = np.array([r.worst for r in reads])
    np.savez_compressed(readers[-1].parent / "readings.npz", ids=np.array(ids), value=vals, dist=dist)
    # one-frame blips among captures in time order (neighbours within 5 s)
    order = sorted((k for k, i in enumerate(ids) if man.get(i) and man[i].time and man[i].root == "captures"),
                   key=lambda k: man[ids[k]].time)
    t = [datetime.fromisoformat(man[ids[k]].time) for k in order]
    blips = []
    for a in range(1, len(order) - 1):
        p, c, n = vals[order[a - 1]], vals[order[a]], vals[order[a + 1]]
        if (t[a + 1] - t[a - 1]).total_seconds() <= 5 and p == n and c != p and min(p, c, n) >= 0:
            blips.append(order[a])
    bad = [k for k, v in enumerate(vals) if v < 0]
    hist = Counter(int(v) for v in vals if v >= 0)
    print(f"read {len(vals) - len(bad)} of {len(vals)} ({100 * (1 - len(bad) / max(1, len(vals))):.2f}%), "
          f"unreadable {len(bad)}, one-frame blips {len(blips)} (of {len(order)} captures in time order)")
    print("values:", dict(sorted(hist.items())))
    tiles = []
    for k in (bad[:30] + blips[:30]):
        c = cv2.resize(np.asarray(crops[k])[40:100, 30:110], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        c = c.copy()
        cv2.putText(c, "?" if vals[k] < 0 else str(vals[k]), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(c)
    if tiles:
        cv2.imwrite(str(readers[-1].parent / "check.png"), sheet(tiles, 10))
        print("examples:", readers[-1].parent / "check.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cluster")
    n = sub.add_parser("name")
    n.add_argument("run")
    n.add_argument("mapping")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "cluster":
        return cmd_cluster()
    if a.cmd == "name":
        return cmd_name(a.run, a.mapping)
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
