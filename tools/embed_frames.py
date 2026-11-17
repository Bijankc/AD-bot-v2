"""Image embeddings for every full frame in the store (for clustering and choosing what to label).

    python tools/embed_frames.py            (new frames only; cached in <ADBOT2_HOME>/data/embeddings/)

MobileNet-V3-small with ImageNet weights (torchvision, BSD; weights cached in <ADBOT2_HOME>/models/torch). Each frame
is mapped to the canonical game area (adbot2.geometry), its model view (1920x1017) fed at 320x170.
Below-normal priority, small batches, a few hundred MB of VRAM at most.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from adbot2 import paths
from adbot2.capture import lower_priority
from adbot2.dataset import load_game_area, load_manifest
from adbot2.geometry import UnknownLayout, model_view

MODEL = "mnv3s_game"  # embeddings of the canonical model view
SIZE = (320, 170)


def emb_path() -> Path:
    return paths.data_dir() / "embeddings" / f"{MODEL}.npz"


def load_embeddings(path: Path | None = None) -> tuple[list[str], np.ndarray]:
    path = path or emb_path()
    if not path.exists():
        return [], np.zeros((0, 576), np.float16)
    z = np.load(path)
    return list(z["ids"]), z["vecs"]


def read_small(rec) -> np.ndarray | None:
    try:
        img = model_view(load_game_area(rec, paths.home()))
    except (UnknownLayout, ValueError, OSError):
        return None
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_model(device):
    os.environ.setdefault("TORCH_HOME", str(paths.home() / "models" / "torch"))
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()  # 576-d pooled features
    return m.eval().to(device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    lower_priority()
    import torch

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ids, vecs = load_embeddings()
    have = set(ids)
    recs = [r for r in load_manifest(paths.manifest_path()).values() if r.full_frame and r.id not in have]
    print(f"{len(have)} cached, {len(recs)} to embed on {device}")
    if not recs:
        return 0
    model = build_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    new_ids, new_vecs = [], []
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool, torch.inference_mode():
        for i in range(0, len(recs), args.batch):
            chunk = recs[i:i + args.batch]
            imgs = list(pool.map(read_small, chunk))
            ok = [(r, im) for r, im in zip(chunk, imgs) if im is not None]
            if not ok:
                continue
            x = torch.from_numpy(np.stack([im for _, im in ok])).to(device).permute(0, 3, 1, 2).float() / 255
            f = model((x - mean) / std)
            f = torch.nn.functional.normalize(f, dim=1)
            new_ids += [r.id for r, _ in ok]
            new_vecs.append(f.cpu().numpy().astype(np.float16))
            if (i // args.batch) % 50 == 0:
                print(f"  {i + len(chunk)}/{len(recs)}  {time.time() - t0:.0f} s", flush=True)
    all_ids = ids + new_ids
    all_vecs = np.concatenate([vecs] + new_vecs) if new_vecs else vecs
    emb_path().parent.mkdir(parents=True, exist_ok=True)
    np.savez(emb_path(), ids=np.array(all_ids), vecs=all_vecs)
    print(f"embedded {len(new_ids)} in {time.time() - t0:.0f} s -> {emb_path()} ({len(all_ids)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
