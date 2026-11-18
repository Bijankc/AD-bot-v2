"""Train one crop head (adbot2.crops.HEADS) on the cached crops; export ONNX; score it on the held-out splits.

    python tools/train_crops.py --head auto

Train = human crop labels of the train split (weight 1) + rule guesses for unlabelled train frames (capped per value,
lower weight; AUTO "off" from the old bot's presses at full weight). No label smoothing: it would cap every
probability below the 0.98 the bot needs to act on OFF. The epoch is chosen on val; test is scored once. Writes <ADBOT2_HOME>/models/crops/<head>/<time>/:
model.pt, model.onnx, meta.json, metrics.json, train.log and preds.npz (the model's guess for every cached crop, used
by the grid page to pre-fill and to pick unsure tiles). Small: seconds to a few minutes, < 0.5 GB VRAM.

For the AUTO head the report also counts ON read as OFF (the error that would make the bot click Auto off), at the
plain argmax and at the "act only when P(off) >= 0.98" threshold the bot will use.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from adbot2 import paths
from adbot2.capture import lower_priority
from adbot2.crops import HEADS, CropCache, CropLabels, auto_press_times, build, model_input, propose
from adbot2.dataset import LabelStore, load_manifest
from adbot2.oldlog import read_all_lines

OFF_THRESHOLD = 0.98


def augment(x: np.ndarray, rng: np.random.Generator, shift: int = 3) -> np.ndarray:
    """N x 3 x H x W in [0,1]: shift a few px, brightness / contrast / small per-channel gain."""
    n = x.shape[0]
    out = np.empty_like(x)
    for k in range(n):
        dx, dy = rng.integers(-shift, shift + 1, 2)
        out[k] = np.roll(x[k], (int(dy), int(dx)), axis=(1, 2))
    b = rng.uniform(-0.12, 0.12, (n, 1, 1, 1))
    c = rng.uniform(0.75, 1.25, (n, 1, 1, 1))
    g = rng.uniform(0.93, 1.07, (n, 3, 1, 1))
    m = out.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip(((out - m) * c + m + b) * g, 0, 1).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True, choices=list(HEADS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rule-cap", type=int, default=1500, help="rule-guessed train crops per value")
    ap.add_argument("--rule-weight", type=float, default=0.3)
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    args = ap.parse_args(argv)
    if sys.stdout is None:
        sys.stdout = sys.stderr = open(os.devnull, "w")
    lower_priority()
    out = paths.home() / "models" / "crops" / args.head / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)

    try:
        return run(args, out, log)
    except Exception:
        import traceback
        log("CRASH\n" + traceback.format_exc())
        return 1


def run(args, out: Path, log) -> int:
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import split_map

    head = HEADS[args.head]
    classes = list(head.classes)
    cidx = {c: k for k, c in enumerate(classes)}
    rng, nrng = random.Random(0), np.random.default_rng(0)
    torch.manual_seed(0)

    ids, crops = CropCache(paths.home() / "cache" / "crops").load(args.head)
    pos = {i: k for k, i in enumerate(ids)}
    man = load_manifest(paths.manifest_path())
    recs = [man[i] for i in ids if i in man]
    split = split_map([r for r in man.values() if r.full_frame])
    human = {i: v for i, v in CropLabels(paths.data_dir() / "labels" / "crops.jsonl").of_head(args.head).items()
             if i in pos}
    screens = LabelStore(paths.screen_labels_path()).all()
    press = auto_press_times(read_all_lines(args.bot_dir)) if args.head == "auto" and args.bot_dir.is_dir() else []
    rule = {r.id: propose(args.head, r, screens.get(r.id) or r.hint, press) for r in recs}

    train = [(i, v, 1.0) for i, v in human.items() if split.get(i) == "train"]
    by_rule: dict[str, list[str]] = defaultdict(list)
    for i, v in rule.items():
        if v and i not in human and split.get(i) == "train":
            by_rule[v].append(i)
    for v, lst in by_rule.items():
        rng.shuffle(lst)
        # "off" from the old bot's Auto presses was right on 15 of 15 checked frames (18 Nov): full weight
        w = 1.0 if (args.head == "auto" and v == "off") else args.rule_weight
        train += [(i, v, w) for i in lst[:args.rule_cap]]
    held = {s: [(i, v) for i, v in human.items() if split.get(i) == s] for s in ("val", "test")}
    log(f"head {args.head}: train {len(train)} ({sum(1 for t in train if t[2] == 1.0)} human, "
        f"{dict(Counter(v for _, v, _ in train))}), val {len(held['val'])}, test {len(held['test'])}")
    if not train:
        log("no training data")
        return 1

    def batch_x(id_list) -> np.ndarray:
        return model_input(np.stack([crops[pos[i]] for i in id_list]), head)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    # sqrt-balanced sampling: rare values are seen more often, without drowning the common ones
    freq = Counter(v for _, v, _ in train)
    sw = np.array([1.0 / np.sqrt(freq[v]) for _, v, _ in train])
    sw /= sw.sum()

    def predict(id_list) -> np.ndarray:
        net.eval()
        res = []
        with torch.no_grad():
            for k in range(0, len(id_list), 512):
                x = torch.from_numpy(batch_x(id_list[k:k + 512])).to(dev)
                res.append(torch.softmax(net(x), 1).cpu().numpy())
        return np.concatenate(res) if res else np.zeros((0, len(classes)))

    def accuracy(items) -> float:
        if not items:
            return float("nan")
        p = predict([i for i, _ in items])
        return float(np.mean([classes[int(k)] == v for k, (_, v) in zip(p.argmax(1), items)]))

    best, best_state, t0 = float("-inf"), None, time.time()
    for ep in range(args.epochs):
        net.train()
        total = 0.0
        for _ in range(max(1, len(train) // args.batch)):
            pick = nrng.choice(len(train), args.batch, p=sw)
            x = augment(batch_x([train[k][0] for k in pick]), nrng)
            y = torch.tensor([cidx[train[k][1]] for k in pick], device=dev)
            w = torch.tensor([train[k][2] for k in pick], device=dev, dtype=torch.float32)
            loss = (F.cross_entropy(net(torch.from_numpy(x).to(dev)), y, reduction="none")
                    * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        va = accuracy(held["val"])
        score = va if va == va else -total  # no val labels yet: keep the last epoch
        if score >= best:
            best, best_state = score, {k: v.detach().clone() for k, v in net.state_dict().items()}
        log(f"epoch {ep + 1}: loss {total:.2f}, val {va:.3f}")
    net.load_state_dict(best_state)
    log(f"trained in {time.time() - t0:.0f} s")

    metrics: dict = {"head": args.head, "classes": classes, "train": len(train),
                     "train_human": sum(1 for t in train if t[2] == 1.0)}
    for s, items in held.items():
        if not items:
            continue
        p = predict([i for i, _ in items])
        pred = [classes[int(k)] for k in p.argmax(1)]
        conf = Counter((v, q) for (_, v), q in zip(items, pred))
        rule_acc = float(np.mean([rule.get(i) == v for i, v in items]))
        m = {"n": len(items), "acc": float(np.mean([q == v for (_, v), q in zip(items, pred)])), "rule_acc": rule_acc,
             "confusion": {f"{a}->{b}": n for (a, b), n in conf.items() if a != b},
             "wrong": [{"id": i, "label": v, "pred": q, "p": float(pp.max())}
                       for (i, v), q, pp in zip(items, pred, p) if q != v][:40]}
        if args.head == "auto":
            on = [k for k, (_, v) in enumerate(items) if v == "on"]
            m["on_read_off"] = int(sum(pred[k] == "off" for k in on))
            m["on_read_off_at_threshold"] = int(sum(p[k, cidx["off"]] >= OFF_THRESHOLD for k in on))
            off = [k for k, (_, v) in enumerate(items) if v == "off"]
            m["off_caught_at_threshold"] = f"{sum(p[k, cidx['off']] >= OFF_THRESHOLD for k in off)}/{len(off)}"
        metrics[s] = m
        log(f"{s}: {m['acc']:.3f} of {m['n']} (rules {rule_acc:.3f}); errors {m['confusion']}"
            + (f"; ON read OFF {m['on_read_off']} (at threshold {m['on_read_off_at_threshold']}), "
               f"OFF caught {m['off_caught_at_threshold']}" if args.head == "auto" else ""))

    # every cached crop: guesses for the grid page
    probs = predict(ids)
    np.savez_compressed(out / "preds.npz", ids=np.array(ids), probs=probs.astype(np.float16))
    torch.save(net.state_dict(), out / "model.pt")
    net_cpu = net.to("cpu").eval()
    w, h = head.size
    torch.onnx.export(net_cpu, torch.zeros(1, 3, h, w), str(out / "model.onnx"), input_names=["crop"],
                      output_names=["logits"], dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    (out / "meta.json").write_text(json.dumps({"head": args.head, "classes": classes}), encoding="utf-8")

    from adbot2.crops import OnnxCropModel
    m = OnnxCropModel(out)
    one = np.asarray(crops[0])
    for _ in range(5):
        m.probs(one)
    t = []
    for _ in range(100):
        a = time.perf_counter()
        m.probs(one)
        t.append(time.perf_counter() - a)
    metrics["ms_cpu"] = 1000 * float(np.median(t))
    pred_counts = Counter(classes[int(k)] for k in probs.argmax(1))
    metrics["pred_counts_all"] = dict(pred_counts)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log(f"ONNX on CPU: {metrics['ms_cpu']:.2f} ms per crop; guesses for all {len(ids)}: {dict(pred_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
