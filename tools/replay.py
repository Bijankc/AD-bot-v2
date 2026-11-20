"""Run the perception step (adbot2.perception) over stored frames in time order and save what it saw.

    python tools/replay.py                    every full frame not replayed yet with the current models
    python tools/replay.py --day 20261118     one capture day only
    python tools/replay.py --redo             replay everything (after new models)

Output: <ADBOT2_HOME>/reports/replay/<models key>.npz with, per frame: id, time, screen probabilities, AUTO
P(on/off/gone), enemy diamond, enemy count (-1 = none), GF banner, menu clock (-1 = none). Used by
tools/smoothing_eval.py and for shadow-mode comparisons with the old bot's log. CPU only, below-normal priority,
decoding in 4 threads. Never committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.capture import lower_priority
from adbot2.dataset import load_game_area, load_manifest
from adbot2.geometry import UnknownLayout
from adbot2.perception import Perception, newest
from adbot2.resources import ram_message, ram_ok


RULE_READERS = "b1"   # version of the readers without a model file (adbot2.buttons); bump to replay again


def models_key(models: Path) -> str:
    """Names of the model runs in use, so a replay is redone when any model changes."""
    parts = []
    for sub, marker in (("screens", "model.onnx"), ("crops/auto", "model.onnx"), ("crops/enemies", "model.onnx"),
                        ("crops/gf_banner", "model.onnx"), ("digits/enemies", "reader.npz"),
                        ("text/menu_clock", "reader.npz")):
        run = newest(models / sub, marker)
        parts.append(run.name if run else "-")
    return "_".join(parts + [RULE_READERS])


def load_replay(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", help="yyyymmdd: only captures of that day")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)
    if sys.stdout is None:
        import os
        sys.stdout = sys.stderr = open(os.devnull, "w")
    lower_priority()
    models = paths.home() / "models"
    out_dir = paths.home() / "reports" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = models_key(models)
    out = out_dir / f"{key}.npz"
    old = load_replay(out) if out.exists() and not args.redo else None
    done = set(old["ids"].tolist()) if old else set()

    recs = [r for r in load_manifest(paths.manifest_path()).values() if r.full_frame and r.time and r.id not in done]
    if args.day:
        recs = [r for r in recs if r.time.replace("-", "").startswith(args.day)]
    recs.sort(key=lambda r: r.time)
    if not recs:
        print(f"nothing new for {key}")
        return 0
    per = Perception(models, gpu=False)
    classes = per.screen.classes if per.screen else []

    def load(r):
        try:
            return load_game_area(r, paths.home())
        except (UnknownLayout, OSError, ValueError):
            return None

    rows: dict[str, list] = {k: [] for k in ("ids", "time", "screen_probs", "auto_probs", "enemies", "count",
                                              "gf", "start", "clock")}
    t0 = time.time()

    def batches():
        """Frames decoded 16 at a time: never more than a few hundred MB in memory (18 Nov: decoding everything at
        once used up the RAM and Roblox crashed)."""
        with ThreadPoolExecutor(4) as ex:
            for k in range(0, len(recs), 16):
                if k % 512 == 0 and not ram_ok():
                    print(ram_message())
                    return
                chunk = recs[k:k + 16]
                yield from zip(chunk, ex.map(load, chunk))

    for r, area in batches():
        if area is None:
            continue
        o = per.observe(area)
        rows["ids"].append(r.id)
        rows["time"].append(r.time)
        rows["screen_probs"].append(o.screen_probs if o.screen_probs is not None else np.zeros(len(classes)))
        rows["auto_probs"].append(o.auto_probs if o.auto_probs is not None else np.zeros(3))
        rows["enemies"].append(bool(o.enemies))
        rows["count"].append(-1 if o.enemy_count is None else o.enemy_count)
        rows["gf"].append(bool(o.gf_banner))
        rows["start"].append(bool(o.start_button))
        rows["clock"].append(-1 if o.menu_clock is None else o.menu_clock[0] * 60 + o.menu_clock[1])
    new = {"ids": np.array(rows["ids"]), "time": np.array(rows["time"]),
           "screen_probs": np.array(rows["screen_probs"], np.float16), "auto_probs": np.array(rows["auto_probs"],
                                                                                              np.float16),
           "enemies": np.array(rows["enemies"]), "count": np.array(rows["count"], np.int16),
           "gf": np.array(rows["gf"]), "start": np.array(rows["start"]), "clock": np.array(rows["clock"], np.int16)}
    if old:
        new = {k: np.concatenate([old[k], new[k]]) for k in new}
        order = np.argsort(new["time"], kind="stable")
        new = {k: v[order] for k, v in new.items()}
    np.savez_compressed(out, **new, classes=np.array(classes))
    (out_dir / "latest.json").write_text(json.dumps({"file": out.name, "classes": classes}), encoding="utf-8")
    n = len(rows["ids"])
    print(f"replayed {n} frames in {time.time() - t0:.0f} s ({1000 * (time.time() - t0) / max(1, n):.0f} ms each) "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
