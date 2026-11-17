"""Crop heads: fixed-position UI read from a fixed box of the canonical frame by a tiny classifier each.

Fixed UI lines up within 1 px at 100% and 150% (bottom- and centre-anchored, see geometry.py), so one box per head
serves both layouts. Boxes are in the canonical 1920x1080 frame (the old bot's coordinates).

Crops of every stored frame are cut once into a cache (tools/cache_crops.py) and labelled on a grid page
(tools/label_screens.py, /crops), pre-filled from simple rules or from the newest crop model. Labels go to
<ADBOT2_HOME>/data/labels/crops.jsonl (append-only, latest row per id+head wins).
"""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from adbot2.dataset import append_jsonl, read_jsonl


@dataclass(frozen=True)
class CropHead:
    name: str
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in the canonical frame
    classes: tuple[str, ...]
    keys: str                         # one hotkey per class, same order
    store: float = 1.0                # scale the crop is cached at
    size: tuple[int, int] = (64, 64)  # model input (w, h)
    help: str = ""

    @property
    def store_wh(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return round((x1 - x0) * self.store), round((y1 - y0) * self.store)


HEADS: dict[str, CropHead] = {h.name: h for h in [
    # The game's auto-battle diamond above the skills. ON pink with "AUTO ON"; OFF dark red with "AUTO"; gone in
    # Dimensions (no AUTO), menus, most results screens and while dead (the "None" sign is lower, on the hotbar).
    # A wrong OFF reading makes the bot click Auto off, so this head must never read ON as OFF.
    CropHead("auto", (680, 790, 800, 910), ("on", "off", "gone"), "ofg",
             help="o = AUTO ON (pink)  f = AUTO off (dark red)  g = no AUTO button"),
    # ENEMIES LEFT diamond (count read separately by the digit reader)
    CropHead("enemies", (1800, 795, 1920, 915), ("shown", "gone"), "sg",
             help="s = ENEMIES LEFT diamond shown  g = not shown"),
    # Group Finder queue banner (raid name, "Raid opens in M:SS", CANCEL), top centre
    CropHead("gf_banner", (760, 85, 1160, 175), ("shown", "gone"), "sg", store=0.5, size=(192, 44),
             help="s = Group Finder banner shown (raid name, CANCEL)  g = not shown"),
]}


def cut(img: np.ndarray, head: CropHead) -> np.ndarray:
    """The head's box from a canonical 1920x1080 BGR frame, at the head's store scale."""
    x0, y0, x1, y1 = head.box
    c = img[y0:y1, x0:x1]
    if head.store != 1.0:
        c = cv2.resize(c, head.store_wh, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(c)


def model_input(crops: np.ndarray, head: CropHead) -> np.ndarray:
    """N x h x w x 3 (or one) cached BGR crops -> N x 3 x H x W float32 in [0, 1] (RGB) at the head's input size."""
    x = crops[None] if crops.ndim == 3 else crops
    w, h = head.size
    if x.shape[1:3] != (h, w):
        x = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in x])
    return np.ascontiguousarray(x[..., ::-1].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)


# ---------------------------------------------------------------- cache


class CropCache:
    """Cached crops: <dir>/<chunk>/ids.json + <head>.npy, one chunk per tools/cache_crops.py run (incremental)."""

    def __init__(self, folder: Path):
        self.folder = folder

    def chunks(self) -> list[Path]:
        return sorted(p for p in self.folder.glob("*") if (p / "ids.json").is_file()) if self.folder.is_dir() else []

    def ids(self) -> list[str]:
        out: list[str] = []
        for c in self.chunks():
            out += json.loads((c / "ids.json").read_text(encoding="utf-8"))["ids"]
        return out

    def skipped(self) -> set[str]:
        out: set[str] = set()
        for c in self.chunks():
            out |= set(json.loads((c / "ids.json").read_text(encoding="utf-8")).get("skipped", []))
        return out

    def load(self, head: str) -> tuple[list[str], np.ndarray]:
        ids, arrays = [], []
        for c in self.chunks():
            f = c / f"{head}.npy"
            if f.is_file():
                ids += json.loads((c / "ids.json").read_text(encoding="utf-8"))["ids"]
                arrays.append(np.load(f, mmap_mode="r"))
        if not arrays:
            w, h = HEADS[head].store_wh
            return [], np.zeros((0, h, w, 3), np.uint8)
        return ids, np.concatenate(arrays) if len(arrays) > 1 else np.asarray(arrays[0])

    def add(self, name: str, ids: list[str], crops: dict[str, np.ndarray], skipped: list[str]) -> Path:
        c = self.folder / name
        c.mkdir(parents=True, exist_ok=True)
        for head, arr in crops.items():
            np.save(c / f"{head}.npy", arr)
        (c / "ids.json").write_text(json.dumps({"ids": ids, "skipped": skipped}), encoding="utf-8")
        return c


# ---------------------------------------------------------------- labels


class CropLabels:
    """Append-only JSONL; the latest row per (id, head) wins."""

    def __init__(self, path: Path):
        self.path = path
        self._latest: dict[tuple[str, str], dict] = {}
        for r in read_jsonl(path):
            self._latest[(r["id"], r["head"])] = r

    def get(self, fid: str, head: str) -> str | None:
        r = self._latest.get((fid, head))
        return r["value"] if r else None

    def set_many(self, head: str, values: dict[str, str], by: str = "human") -> None:
        cls = HEADS[head].classes
        at = datetime.now().isoformat(timespec="seconds")
        rows = [{"id": i, "head": head, "value": v, "by": by, "at": at} for i, v in values.items() if v in cls]
        append_jsonl(self.path, rows)
        for r in rows:
            self._latest[(r["id"], head)] = r

    def of_head(self, head: str) -> dict[str, str]:
        return {i: r["value"] for (i, h), r in self._latest.items() if h == head}


# ---------------------------------------------------------------- weak proposals

HUD_SCREENS = {"fight", "respawn", "dead", "revive"}
NO_AUTO_RULES = ("titan_", "dim_wait")   # Dimensions have no AUTO button
AUTO_OFF_BEFORE = timedelta(seconds=2.5)  # Auto is off just before the old bot presses it


def auto_press_times(lines) -> list[datetime]:
    """Times the old bot pressed Auto (or logged "Auto: off" in older logs), sorted."""
    return sorted(ln.time for ln in lines if ln.text.startswith(("Auto: off",)) or "Auto pressed (try" in ln.text)


def propose(head: str, rec, screen: str | None, press_times: list[datetime] | None = None) -> str | None:
    """A rule-based guess for a crop value (pre-fills the grid until a crop model exists). None = no guess."""
    if screen is None or screen in ("unsure", "other"):
        return None
    if head == "auto":
        if screen not in HUD_SCREENS:
            return "gone"
        if screen in ("dead", "revive") or (rec.rule or "").startswith(NO_AUTO_RULES):
            return "gone"
        if press_times and rec.time:
            t = datetime.fromisoformat(rec.time)
            k = bisect.bisect_left(press_times, t)
            if k < len(press_times) and press_times[k] - t <= AUTO_OFF_BEFORE:
                return "off"
        return "on"
    if head == "enemies":
        return "shown" if screen in HUD_SCREENS - {"revive"} else "gone"
    if head == "gf_banner":
        return "shown" if screen == "raid_world" or rec.rule == "gf_queued" else "gone"
    return None


# ---------------------------------------------------------------- picking a grid page


@dataclass
class Candidate:
    id: str
    time: datetime | None
    guess: str | None         # pre-filled value
    conf: float               # model confidence in the guess (1.0 for rule guesses)
    rule_guess: str | None = None


def pick_page(cands: list[Candidate], n: int, gap_s: float = 4.0, seed: int = 0) -> list[Candidate]:
    """Up to n unlabelled crops to show on one grid page: first model/rule disagreements and the least confident,
    then a balanced share of every guessed value (rare values first), at least gap_s apart in time."""
    rng = random.Random(seed)
    pool = cands[:]
    rng.shuffle(pool)
    chosen: list[Candidate] = []
    times: list[datetime] = []

    def take(c: Candidate) -> bool:
        if c.time and any(abs((c.time - u).total_seconds()) < gap_s for u in times):
            return False
        chosen.append(c)
        if c.time:
            times.append(c.time)
        return True

    hard = sorted((c for c in pool if (c.rule_guess and c.guess != c.rule_guess) or c.conf < 0.9),
                  key=lambda c: c.conf)
    for c in hard:
        if len(chosen) >= n // 2:
            break
        take(c)
    picked = {c.id for c in chosen}
    by_value: dict[str | None, list[Candidate]] = {}
    for c in pool:
        if c.id not in picked:
            by_value.setdefault(c.guess, []).append(c)
    order = sorted(by_value, key=lambda v: len(by_value[v]))
    while len(chosen) < n and any(by_value.values()):
        for v in order:
            while by_value[v]:
                if take(by_value[v].pop()):
                    break
            if len(chosen) >= n:
                break
    return chosen


# ---------------------------------------------------------------- model


def build(n_classes: int):
    """A tiny CNN (~0.1 M weights) for one head; trained from scratch, so no pretrained weights are involved."""
    import torch.nn as nn

    def blk(i: int, o: int) -> list:
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    return nn.Sequential(*blk(3, 16), nn.MaxPool2d(2), *blk(16, 32), nn.MaxPool2d(2), *blk(32, 64), nn.MaxPool2d(2),
                         *blk(64, 96), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, n_classes))


class OnnxCropModel:
    """An exported head (model.onnx + meta.json in one folder), run with onnxruntime."""

    def __init__(self, folder: Path, gpu: bool = False):
        import onnxruntime as ort

        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.head = HEADS[meta["head"]]
        self.classes: list[str] = meta["classes"]
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(folder / "model.onnx"), providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    def probs(self, crops: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {self.inp: model_input(crops, self.head)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def read(self, frame: np.ndarray) -> tuple[str, float]:
        """(value, probability) from a canonical 1920x1080 frame."""
        p = self.probs(cut(frame, self.head))[0]
        return self.classes[int(p.argmax())], float(p.max())
