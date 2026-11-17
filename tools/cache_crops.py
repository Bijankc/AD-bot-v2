"""Cut every crop head's box (adbot2.crops.HEADS) from every stored full frame not yet cached.

    python tools/cache_crops.py            new frames only (one chunk per run)
    python tools/cache_crops.py --rebuild  drop the cache first (after a box changes)

Each frame is decoded once and mapped to the canonical frame (100%, 150% and old-bot shots alike). Frames of an unknown
layout (mid-resize) are recorded as skipped. Below-normal priority, 6 decoding threads.
Cache: <ADBOT2_HOME>/cache/crops/<chunk>/ (never committed).
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from adbot2 import paths
from adbot2.capture import lower_priority
from adbot2.crops import HEADS, CropCache, cut
from adbot2.dataset import load_game_area, load_manifest
from adbot2.geometry import UnknownLayout


def crop_cache() -> CropCache:
    return CropCache(paths.home() / "cache" / "crops")


def cut_all(rec) -> dict[str, np.ndarray] | None:
    try:
        img = load_game_area(rec, paths.home()).image
    except (UnknownLayout, OSError, ValueError):
        return None
    return {name: cut(img, h) for name, h in HEADS.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    lower_priority()
    cache = crop_cache()
    if args.rebuild and cache.folder.is_dir():
        shutil.rmtree(cache.folder)
    done = set(cache.ids()) | cache.skipped()
    recs = [r for r in load_manifest(paths.manifest_path()).values() if r.full_frame and r.id not in done]
    if not recs:
        print("nothing new")
        return 0
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(cut_all, recs))
    ids = [r.id for r, c in zip(recs, results) if c is not None]
    skipped = [r.id for r, c in zip(recs, results) if c is None]
    crops = {name: np.stack([c[name] for c in results if c is not None]) if ids else
             np.zeros((0, h.store_wh[1], h.store_wh[0], 3), np.uint8) for name, h in HEADS.items()}
    where = cache.add(f"{datetime.now():%Y%m%d_%H%M%S}", ids, crops, skipped)
    mb = sum(a.nbytes for a in crops.values()) / 1e6
    print(f"{len(ids)} frames cached ({len(skipped)} skipped) in {time.time() - t0:.0f} s, {mb:.0f} MB -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
