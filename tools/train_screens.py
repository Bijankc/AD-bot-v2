"""Train the screen classifier, export it to ONNX and measure its speed.

    pythonw tools/train_screens.py            (detached; progress in <ADBOT2_HOME>/models/screens/<time>/train.log)

Training frames: the user's labels on train-split sessions (weight 1) plus weak hints from the old bot's log on the same
sessions (weight --hint-weight, at most --hint-cap per class, never a frame within 3 min of a val/test frame: the split's
guard band does that). Val = the user's labels on val sessions; the best epoch by val accuracy is kept. Test is left to
tools/evaluate.py --predictor model.

Farming first: below-normal priority, batch 32 with mixed precision (~1.5 GB VRAM), the old bot's log is checked after
every epoch and the job stops if Roblox freezes or restarts more than usual.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.capture import lower_priority
from adbot2.dataset import LabelStore, load_game_area, load_manifest
from adbot2.geometry import UnknownLayout
from adbot2.resources import ram_message, ram_ok
from adbot2.screen_model import INPUT_H, INPUT_W, build, to_tensor_array, view_small
from adbot2.screens import Screen

CLASSES = [s.value for s in Screen if s is not Screen.UNSURE]
TROUBLE = re.compile(r"restarting Roblox(?!: preventive)|not responding|screen not recognised")


def cache_dir() -> Path:
    return paths.home() / "cache" / f"screen{INPUT_W}x{INPUT_H}"


def small_of(rec) -> np.ndarray | None:
    """The record's small model view, cached on disk (None for crops / unknown layouts)."""
    f = cache_dir() / f"{rec.id}.npy"
    if f.exists():
        return np.load(f)
    try:
        v = view_small(load_game_area(rec, paths.home()))
    except (UnknownLayout, OSError, ValueError):
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    return v


def load_smalls(recs) -> dict[str, np.ndarray]:
    """Small views, decoded 256 frames at a time; stops early (fewer frames) under the RAM floor."""
    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(6) as ex:
        for k in range(0, len(recs), 256):
            if not ram_ok():
                raise SystemExit(ram_message())
            chunk = recs[k:k + 256]
            out.update(zip((r.id for r in chunk), ex.map(small_of, chunk)))
    return {k: v for k, v in out.items() if v is not None}


def trouble_lines(bot_dir: Path, since: datetime) -> int:
    """Lines in the old bot's raid.log since `since` that show Roblox trouble (quick open/close)."""
    f = bot_dir / "raid.log"
    if not f.exists():
        return 0
    with open(f, encoding="utf-8", errors="replace") as fh:
        tail = fh.readlines()[-400:]
    n = 0
    for ln in tail:
        try:
            t = datetime.strptime(f"{since.year}.{ln[:18]}", "%Y.%d.%m %H:%M:%S.%f")  # "17.11 18:13:43.878"
        except ValueError:
            continue
        if t >= since and TROUBLE.search(ln):
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--hint-cap", type=int, default=600)
    ap.add_argument("--hint-weight", type=float, default=0.5)
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--max-trouble", type=int, default=3, help="stop after this many trouble lines in the old bot's log")
    args = ap.parse_args(argv)

    if sys.stdout is None:  # pythonw: libraries that print (e.g. torch's weight download) need somewhere to write
        sys.stdout = sys.stderr = open(os.devnull, "w")
    lower_priority()
    out =paths.home() / "models" / "screens" / f"{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        logf.write(line + "\n")
        logf.flush()
        if sys.stdout is not None:
            print(line, flush=True)

    try:
        return run(args, out, log)
    except Exception:
        import traceback
        log("CRASH\n" + traceback.format_exc())
        return 1


def run(args, out: Path, log) -> int:
    os.environ.setdefault("TORCH_HOME", str(paths.home() / "models" / "torch"))
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import split_map

    started = datetime.now()
    rng = random.Random(0)
    man = load_manifest(paths.manifest_path())
    recs = [r for r in man.values() if r.full_frame]
    split = split_map(recs)
    labels = {k: v for k, v in LabelStore(paths.screen_labels_path()).all().items()
              if k in man and v != "unsure" and man[k].full_frame}

    train = [(i, lab, 1.0) for i, lab in labels.items() if split.get(i) == "train"]
    val = [(i, lab) for i, lab in labels.items() if split.get(i) == "val"]
    by_hint: dict[str, list[str]] = {}
    for r in recs:
        if r.hint and r.id not in labels and split.get(r.id) == "train" and r.hint in CLASSES:
            by_hint.setdefault(r.hint, []).append(r.id)
    for c, ids in by_hint.items():
        rng.shuffle(ids)
        train += [(i, c, args.hint_weight) for i in ids[:args.hint_cap]]
    log(f"train: {sum(w == 1.0 for _, _, w in train)} labelled + {sum(w != 1.0 for _, _, w in train)} hinted; "
        f"val: {len(val)} labelled")
    log("train classes: " + ", ".join(f"{c} {n}" for c, n in Counter(c for _, c, _ in train).most_common()))

    t0 = time.time()
    smalls = load_smalls([man[i] for i in {i for i, _, _ in train} | {i for i, _ in val}])
    log(f"views ready in {time.time() - t0:.0f} s ({len(smalls)} frames)")
    train = [t for t in train if t[0] in smalls]
    val = [v for v in val if v[0] in smalls]
    cidx = {c: k for k, c in enumerate(CLASSES)}

    xt = np.stack([smalls[i] for i, _, _ in train])
    yt = torch.tensor([cidx[c] for _, c, _ in train])
    wt = torch.tensor([w for _, _, w in train], dtype=torch.float32)
    xv = torch.from_numpy(to_tensor_array(np.stack([smalls[i] for i, _ in val]))) if val else None
    yv = torch.tensor([cidx[c] for _, c in val])

    # sampling: each class drawn ~ sqrt of its size, so rare screens are seen often without being memorised
    freq = Counter(yt.tolist())
    samp_w = torch.tensor([1.0 / freq[int(y)] ** 0.5 for y in yt])
    steps = max(1, len(train) // args.batch)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(0.35) if dev.type == "cuda" else None
    net = build(len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def batch_tensor(idx: torch.Tensor, augment: bool) -> torch.Tensor:
        x = torch.from_numpy(xt[idx.numpy()]).to(dev).flip(-1).permute(0, 3, 1, 2).float() / 255  # BGR -> RGB
        if augment:
            n = x.shape[0]
            x = x * (1 + 0.3 * (torch.rand(n, 1, 1, 1, device=dev) - 0.5)) + 0.16 * (torch.rand(n, 1, 1, 1, device=dev) - 0.5)
            dx, dy = random.randint(-8, 8), random.randint(-6, 6)   # small shifts (window placement, 150% layout)
            x = torch.roll(x, (dy, dx), (2, 3))
            x = x.clamp(0, 1)
        return (x - mean) / std

    best, best_ep = -1.0, -1
    for ep in range(args.epochs):
        net.train()
        order = torch.multinomial(samp_w, steps * args.batch, replacement=True)
        tl, t0 = 0.0, time.time()
        for s in range(steps):
            idx = order[s * args.batch:(s + 1) * args.batch]
            x = batch_tensor(idx, True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = (F.cross_entropy(net(x), yt[idx].to(dev), label_smoothing=0.05, reduction="none")
                        * wt[idx].to(dev)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += float(loss.detach())
        acc = float("nan")
        if xv is not None:
            net.eval()
            with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
                pv = torch.cat([net(xv[k:k + 64].to(dev)).argmax(1).cpu() for k in range(0, len(xv), 64)])
            acc = float((pv == yv).float().mean())
            if acc > best:
                best, best_ep = acc, ep
                torch.save(net.state_dict(), out / "model.pt")
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        log(f"epoch {ep + 1}/{args.epochs}: loss {tl / steps:.3f}, val {acc:.1%}, {time.time() - t0:.0f} s, "
            f"VRAM peak {mem:.2f} GB")
        trouble = trouble_lines(args.bot_dir, started)
        if trouble > args.max_trouble:
            log(f"STOP: {trouble} trouble lines in the old bot's log since the start")
            break
    if best_ep < 0:
        torch.save(net.state_dict(), out / "model.pt")
    log(f"best val {best:.1%} at epoch {best_ep + 1}")

    # export + speed
    net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    net.eval().float().cpu()
    (out / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    torch.onnx.export(net, torch.zeros(1, 3, INPUT_H, INPUT_W), str(out / "model.onnx"), input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    speed = measure(out, log)
    (out / "metrics.json").write_text(json.dumps({
        "best_val": best, "best_epoch": best_ep + 1, "train": len(train), "val": len(val), "classes": CLASSES,
        "input": [INPUT_W, INPUT_H], "args": {k: str(v) for k, v in vars(args).items()}, **speed}, indent=1),
        encoding="utf-8")
    log(f"done: {out}")
    return 0


def measure(folder: Path, log) -> dict:
    """Median ms per frame: preprocessing (1920x1080 -> small view) and the model on CPU and GPU, batch 1."""
    from adbot2.screen_model import OnnxScreenModel

    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), np.uint8)
    res = {}
    t = []
    for _ in range(30):
        a = time.perf_counter()
        small = view_small(frame)
        t.append(time.perf_counter() - a)
    res["ms_preprocess"] = 1000 * float(np.median(t))
    for name, gpu in (("cpu", False), ("gpu", True)):
        try:
            m = OnnxScreenModel(folder, gpu=gpu)
        except Exception as e:  # noqa: BLE001
            log(f"speed {name}: {e}")
            continue
        for _ in range(5):
            m.probs(small)
        t = []
        for _ in range(50):
            a = time.perf_counter()
            m.probs(small)
            t.append(time.perf_counter() - a)
        res[f"ms_{name}"] = 1000 * float(np.median(t))
        res[f"providers_{name}"] = m.sess.get_providers()
    log("speed: " + ", ".join(f"{k} {v:.1f}" for k, v in res.items() if k.startswith("ms_")))
    return res


if __name__ == "__main__":
    raise SystemExit(main())
