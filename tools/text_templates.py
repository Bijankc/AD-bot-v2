"""Build and check a text-line reader (adbot2.textline) for one field, from the cached text bands.

    python tools/text_templates.py cluster FIELD              group every glyph; writes clusters.png to look at
    python tools/text_templates.py split FIELD RUN C K        split mixed cluster C into K sub-clusters (C.0 .. C.K-1)
    python tools/text_templates.py name FIELD RUN "0:a,1:7,4.2:9"   cluster -> text ("-" = skip); fits reader.npz
    python tools/text_templates.py check FIELD                read every candidate band; report and consistency check
    python tools/text_templates.py selftrain FIELD            label clock digits from the clock itself (see below)

selftrain: the newest reader finds the hour's opening second (the most common one); every band's correct countdown
then follows from its capture time. Digit glyphs whose value is certain from 1.5 s early to LAG s late are labelled with
it, a nearest-example reader is fitted on the first half of the days and scored on the rest, then refitted on all.

Candidates are bands that look like text on a dark bar (dark median, a few % bright pixels). The check uses the clock
itself: a countdown read at time t means the raid opens at second (t + left) mod 3600 of the hour, so readings of one
server must agree. Readings whose opening second is more than 3 s away from the median of their hour are listed.
Runs live in <ADBOT2_HOME>/models/text/<field>/<time>/ (never committed).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.crops import CropCache
from adbot2.dataset import load_manifest
from adbot2.textline import FIELDS, TextReader, on_dark_bar, segment

K = 64
LAG = 6  # s the game's countdown text may lag behind the real countdown


def runs_dir(field: str) -> Path:
    return paths.home() / "models" / "text" / field


def candidates(field: str) -> tuple[list[str], np.ndarray]:
    f = FIELDS[field]
    ids, crops = CropCache(paths.home() / "cache" / "crops").load(field)
    keep = []
    for k in range(len(ids)):
        if on_dark_bar(np.asarray(crops[k]), f):
            keep.append(k)
    return [ids[k] for k in keep], np.asarray(crops[keep])


def tile_sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    tiles = [cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=128) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def cmd_cluster(field: str) -> int:
    f = FIELDS[field]
    ids, crops = candidates(field)
    gl = [g for c in crops for _, g in segment(c, f)]
    if len(gl) < K:
        sys.exit(f"only {len(gl)} glyphs in {len(ids)} candidate bands")
    x = np.stack([g.reshape(-1) for g in gl]).astype(np.float32)
    cv2.setRNGSeed(0)
    _, lab, _ = cv2.kmeans(x, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05), 4,
                           cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    run = runs_dir(field) / f"{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "clusters.npz", x=x.reshape(-1, *f.canvas), lab=lab)
    ch, cw = f.canvas
    rows = []
    for c in range(K):
        mem = np.where(lab == c)[0][:14]
        tiles = [cv2.resize((x[i].reshape(ch, cw) * 255).astype(np.uint8), None, fx=2, fy=2,
                            interpolation=cv2.INTER_NEAREST) for i in mem]
        tiles += [np.zeros((ch * 2, cw * 2), np.uint8)] * (14 - len(tiles))
        name = np.zeros((ch * 2 + 2, 80), np.uint8)
        cv2.putText(name, f"{c}:{int((lab == c).sum())}", (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
        rows.append(np.hstack([name, tile_sheet(tiles, 14)]))
    half = (K + 1) // 2
    left, right = np.vstack(rows[:half]), np.vstack(rows[half:])
    if right.shape[0] < left.shape[0]:
        right = cv2.copyMakeBorder(right, 0, left.shape[0] - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite(str(run / "clusters.png"), np.hstack([left, np.full((left.shape[0], 6), 255, np.uint8), right]))
    print(f"{len(gl)} glyphs from {len(ids)} candidate bands -> {run / 'clusters.png'}")
    return 0


def sub_labels(x: np.ndarray, lab: np.ndarray, c: int, k: int) -> np.ndarray:
    """Deterministic sub-cluster ids (0..k-1) for the members of cluster c, in member order."""
    mem = np.where(lab == c)[0]
    cv2.setRNGSeed(c)
    _, sub, _ = cv2.kmeans(x[mem].reshape(len(mem), -1).astype(np.float32), k, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05), 6, cv2.KMEANS_PP_CENTERS)
    return sub.ravel()


def keys_of(x: np.ndarray, lab: np.ndarray, mapping: str) -> list[str]:
    """Cluster key per glyph: "C", or "C.k" for clusters split in the mapping."""
    splits: dict[int, int] = {}
    for p in mapping.split(","):
        a = p.split(":", 1)[0]
        if "." in a:
            c, k = a.split(".")
            splits[int(c)] = max(splits.get(int(c), 0), int(k) + 1)
    keys = [str(int(c)) for c in lab]
    for c, k in splits.items():
        mem = np.where(lab == c)[0]
        for i, sk in zip(mem, sub_labels(x, lab, c, k)):
            keys[i] = f"{c}.{int(sk)}"
    return keys


def cmd_split(field: str, run: str, c: int, k: int) -> int:
    f = FIELDS[field]
    folder = runs_dir(field) / run
    with np.load(folder / "clusters.npz") as z:
        x, lab = z["x"], z["lab"]
    sub = sub_labels(x, lab, c, k)
    mem = np.where(lab == c)[0]
    ch, cw = f.canvas
    rows = []
    for j in range(k):
        pick = mem[sub == j][:14]
        tiles = [cv2.resize((x[i] * 255).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                 for i in pick] + [np.zeros((ch * 2, cw * 2), np.uint8)] * (14 - len(pick))
        name = np.zeros((ch * 2 + 2, 80), np.uint8)
        cv2.putText(name, f"{c}.{j}:{int((sub == j).sum())}", (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        rows.append(np.hstack([name, tile_sheet(tiles, 14)]))
    cv2.imwrite(str(folder / f"split_{c}.png"), np.vstack(rows))
    print(folder / f"split_{c}.png")
    return 0


def cmd_name(field: str, run: str, mapping: str) -> int:
    folder = runs_dir(field) / run
    with np.load(folder / "clusters.npz") as z:
        x, lab = z["x"], z["lab"]
    m = dict(p.split(":", 1) for p in mapping.split(","))
    keys = keys_of(x, lab, mapping)
    keep = [i for i, key in enumerate(keys) if m.get(key, "-") != "-"]
    reader = TextReader.fit(FIELDS[field], [x[i] for i in keep], [m[keys[i]] for i in keep])
    reader.save(folder / "reader.npz")
    print(f"reader from {len(keep)} glyphs, characters {''.join(reader.chars)}, max distance {reader.max_dist:.2f}")
    return 0


def cmd_check(field: str) -> int:
    readers = sorted(runs_dir(field).glob("*/reader.npz"))
    if not readers:
        sys.exit("no reader yet: run cluster, then name")
    reader = TextReader.load(readers[-1])
    ids, crops = candidates(field)
    man = load_manifest(paths.manifest_path())
    reads = [reader.read_crop(c) for c in crops]
    got = [(i, r) for i, r in zip(ids, reads) if r.value]
    texts = Counter(r.text for r in reads if not r.value)
    print(f"{len(ids)} candidate bands: {len(got)} read a clock, {len(ids) - len(got)} not")
    print("most common unread texts:", texts.most_common(8))
    # consistency: the opening second of the hour implied by each reading, grouped by hour of capture
    by_hour: dict[str, list] = defaultdict(list)
    for i, r in got:
        rec = man.get(i)
        if not rec or not rec.time:
            continue
        t = datetime.fromisoformat(rec.time)
        left = r.value[0] * 60 + r.value[1]
        ofs = (t.minute * 60 + t.second + t.microsecond / 1e6 + left) % 3600
        by_hour[f"{t:%Y%m%d_%H}"].append((ofs, i, r, t))
    bad, total = [], 0
    for hour, lst in sorted(by_hour.items()):
        med = float(np.median([o for o, *_ in lst]))
        for o, i, r, t in lst:
            total += 1
            d = min(abs(o - med), 3600 - abs(o - med))
            if d > 3:
                bad.append((hour, d, i, r, t))
        print(f"  {hour}: {len(lst)} readings, opens at :{int(med) // 60:02}:{int(med) % 60:02}")
    print(f"consistency: {total - len(bad)} of {total} readings within 3 s of their hour's opening time")
    for hour, d, i, r, t in bad[:15]:
        print(f"  off by {d:.0f} s: {t:%d.%m %H:%M:%S} read {r.text!r}")
    np.savez_compressed(readers[-1].parent / "readings.npz", ids=np.array([i for i, _ in got]),
                        text=np.array([r.text for _, r in got]))
    tiles = []
    for (hour, d, i, r, t) in bad[:20]:
        c = np.asarray(crops[ids.index(i)])
        tiles.append(cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    if tiles:
        cv2.imwrite(str(readers[-1].parent / "check.png"), tile_sheet(tiles, 2))
        print("examples:", readers[-1].parent / "check.png")
    return 0


def minute_digits(field: str) -> int:
    """2 for MM:SS fields, 1 for M:SS (from the field's pattern)."""
    return 2 if FIELDS[field].pattern.startswith(r"(\d\d)") else 1


def clock_glyphs(reader: TextReader, crop: np.ndarray, nm: int) -> tuple[list[np.ndarray], list[bool]] | None:
    """Glyphs of the trailing clock (nm minute digits, colon, 2 second digits; the last minute digit may be drawn
    touching the colon as "2:") and, per glyph, whether it is that merged "2:". None if the end is no clock."""
    segs = [g for _, g in segment(crop, reader.f)]
    texts = []
    for g in segs:
        d = np.linalg.norm(reader.t - g.reshape(1, -1), axis=1)
        texts.append(reader.chars[int(d.argmin())])
    if len(segs) >= nm + 3 and texts[-3] == ":" and all(t[0].isdigit() and ":" not in t for t in texts[-3 - nm:-3]):
        return segs[-3 - nm:-3] + segs[-2:], [False] * (nm + 2)
    if len(segs) >= nm + 2 and texts[-3] == "2:" and all(t[0].isdigit() and ":" not in t for t in texts[-2 - nm:-3]):
        return segs[-2 - nm:-3] + [segs[-3]] + segs[-2:], [False] * (nm - 1) + [True, False, False]
    return None


def cmd_selftrain(field: str) -> int:
    f = FIELDS[field]
    readers = sorted(runs_dir(field).glob("*/reader.npz"))
    reader = TextReader.load(readers[-1])
    ids, crops = candidates(field)
    man = load_manifest(paths.manifest_path())
    secs, offsets = {}, []
    for i, c in zip(ids, crops):
        rec = man.get(i)
        if not rec or not rec.time:
            continue
        t = datetime.fromisoformat(rec.time)
        secs[i] = t.minute * 60 + t.second + t.microsecond / 1e6
        r = reader.read_crop(c)
        if r.value:
            offsets.append(round(secs[i] + r.value[0] * 60 + r.value[1]) % 3600)
    ofs = Counter(offsets).most_common(1)[0][0]
    near = sum(min(abs(o - ofs), 3600 - abs(o - ofs)) <= 2 for o in offsets)
    print(f"opening second of the hour: :{ofs // 60:02}:{ofs % 60:02} ({near} of {len(offsets)} readings within 2 s)")

    nm = minute_digits(field)

    def clock_text(v: float) -> str:
        return f"{int(v) // 60:0{nm}}:{int(v) % 60:02}"

    def truth(i: str) -> list[str]:
        """Possible clock strings: from 1.5 s early to LAG s late (the game's text can lag ~5 s)."""
        exp = (ofs - secs[i]) % 3600
        vals = (exp + d / 2 for d in range(-3, 2 * LAG + 1))
        return sorted({clock_text(v) for v in vals if 0 <= v < 3600})

    days = sorted({man[i].time[:10] for i in secs})
    train_days = set(days[:max(1, len(days) // 2)])
    gl, ch, day_of = [], [], []
    for i, c in zip(ids, crops):
        if i not in secs:
            continue
        got = clock_glyphs(reader, c, nm)
        if not got:
            continue
        segs, merged = got
        opts = [o for o in truth(i) if len(o) == nm + 3]
        if not opts:
            continue
        digit_pos = [k for k, chr_ in enumerate(opts[0]) if chr_ != ":"]
        for g, j, mg in zip(segs, digit_pos, merged):
            vals = {o[j] for o in opts}
            if len(vals) != 1:
                continue  # not certain at this timing
            v = vals.pop()
            if mg:
                if v != "2":
                    continue
                v = "2:"
            gl.append(g)
            ch.append(v)
            day_of.append(man[i].time[:10])
    # non-digit shapes (letters, colon) from the current reader keep letters from passing as digits
    extra = [(reader.t[k].reshape(f.canvas), c) for k, c in enumerate(reader.chars) if not c[0].isdigit()]
    print(f"labelled digit glyphs: {len(gl)} ({dict(sorted(Counter(ch).items()))}); train days {sorted(train_days)}")

    def build(sel) -> TextReader:
        g = [gl[k] for k in sel] + [e for e, _ in extra]
        c = [ch[k] for k in sel] + [c for _, c in extra]
        return TextReader.from_examples(f, g, c, reader.max_dist)

    def score(r: TextReader, test_ids) -> tuple[int, int, int]:
        ok = bad = none = 0
        for i in test_ids:
            rr = r.read_crop(crops[ids.index(i)])
            if not rr.value:
                none += 1
                continue
            left = rr.value[0] * 60 + rr.value[1]
            exp = (ofs - secs[i]) % 3600
            if -2 <= left - exp <= LAG:
                ok += 1
            else:
                bad += 1
        return ok, bad, none

    test_ids = [i for i in ids if i in secs and man[i].time[:10] not in train_days]
    held = build([k for k, d in enumerate(day_of) if d in train_days])
    ok, bad, none = score(held, test_ids)
    # the decision the bot makes: a raid time is accepted when two readings within 2 min agree within 2 s
    seq = []
    for i in sorted(test_ids, key=lambda i: man[i].time):
        rr = held.read_crop(crops[ids.index(i)])
        if rr.value:
            seq.append((datetime.fromisoformat(man[i].time), (secs[i] + rr.value[0] * 60 + rr.value[1]) % 3600))
    accepted = []
    for (t1, o1), (t2, o2) in zip(seq, seq[1:]):
        if (t2 - t1).total_seconds() <= 120 and min(abs(o1 - o2), 3600 - abs(o1 - o2)) <= 2:
            accepted.append(o2)
    wrong = [o for o in accepted if not -LAG - 2 <= ((o - ofs + 1800) % 3600 - 1800) <= 2]
    print(f"decision on held-out days: {len(accepted)} raid times accepted (two agreeing readings), "
          f"{len(wrong)} wrong {[f':{int(o) // 60:02}:{int(o) % 60:02}' for o in wrong]}")
    old_ok, old_bad, old_none = score(reader, test_ids)
    print(f"held-out days ({len(test_ids)} bands): new reader {ok} right, {bad} wrong, {none} no reading; "
          f"old reader {old_ok} right, {old_bad} wrong, {old_none} no reading")
    final = build(range(len(gl)))
    out = runs_dir(field) / f"{datetime.now():%Y%m%d_%H%M%S}_self"
    final.save(out / "reader.npz")
    print(f"saved {out / 'reader.npz'} ({len(final.chars)} examples)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cluster", "check", "selftrain"):
        p = sub.add_parser(name)
        p.add_argument("field", choices=list(FIELDS))
    sp = sub.add_parser("split")
    sp.add_argument("field", choices=list(FIELDS))
    sp.add_argument("run")
    sp.add_argument("c", type=int)
    sp.add_argument("k", type=int)
    n = sub.add_parser("name")
    n.add_argument("field", choices=list(FIELDS))
    n.add_argument("run")
    n.add_argument("mapping")
    a = ap.parse_args(argv)
    if a.cmd == "cluster":
        return cmd_cluster(a.field)
    if a.cmd == "name":
        return cmd_name(a.field, a.run, a.mapping)
    if a.cmd == "selftrain":
        return cmd_selftrain(a.field)
    if a.cmd == "split":
        return cmd_split(a.field, a.run, a.c, a.k)
    return cmd_check(a.field)


if __name__ == "__main__":
    raise SystemExit(main())
