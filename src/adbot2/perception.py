"""One frame -> one Observation: the screen classifier plus every fixed-UI reader, with timings.

Readers run only where they make sense (the enemy count when its diamond is shown, the menu clock on the modes
menu), so a frame costs the screen model plus well under a millisecond per crop head. Models are the newest trained
ones under <ADBOT2_HOME>/models/ (never committed); a missing model just leaves its fields empty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adbot2.crops import HEADS, OnnxCropModel, cut
from adbot2.digits import DigitReader
from adbot2.geometry import GameArea
from adbot2.screen_model import OnnxScreenModel, view_small
from adbot2.textline import FIELDS, TextReader, cut_field, on_dark_bar

AUTO_OFF_ACT = 0.98   # P(off) needed before anything may treat Auto as off


@dataclass
class Observation:
    screen: str | None = None
    screen_p: float = 0.0
    screen_probs: np.ndarray | None = None
    auto: str | None = None          # on | off | gone
    auto_p_off: float = 0.0
    auto_probs: np.ndarray | None = None   # in the head's class order (on, off, gone)
    enemies: bool | None = None      # ENEMIES LEFT diamond shown
    enemy_count: int | None = None
    gf_banner: bool | None = None
    menu_clock: tuple[int, int] | None = None   # (minutes, seconds) on the RAID card
    ms: dict[str, float] = field(default_factory=dict)

    @property
    def auto_is_off(self) -> bool:
        """Safe test for "Auto is off": only when the reader is very sure (ON read as OFF would switch Auto off)."""
        return self.auto_p_off >= AUTO_OFF_ACT


def newest(folder: Path, marker: str) -> Path | None:
    """Newest sub-folder (by name = time) of `folder` that contains `marker`."""
    runs = sorted(p.parent for p in folder.glob(f"*/{marker}"))
    return runs[-1] if runs else None


class Perception:
    def __init__(self, models: Path, gpu: bool = True):
        self.screen = None
        if (run := newest(models / "screens", "model.onnx")) is not None:
            self.screen = OnnxScreenModel(run, gpu=gpu)
        self.heads: dict[str, OnnxCropModel] = {}
        for name in HEADS:
            if (run := newest(models / "crops" / name, "model.onnx")) is not None:
                self.heads[name] = OnnxCropModel(run)
        self.digits = None
        if (run := newest(models / "digits" / "enemies", "reader.npz")) is not None:
            self.digits = DigitReader.load(run / "reader.npz")
        self.menu_clock = None
        if (run := newest(models / "text" / "menu_clock", "reader.npz")) is not None:
            self.menu_clock = TextReader.load(run / "reader.npz")

    def observe(self, area: GameArea | np.ndarray) -> Observation:
        img = area.image if isinstance(area, GameArea) else area
        o = Observation()
        t0 = time.perf_counter()
        if self.screen is not None:
            p = self.screen.probs(view_small(img))[0]
            k = int(p.argmax())
            o.screen, o.screen_p, o.screen_probs = self.screen.classes[k], float(p[k]), p
        t1 = time.perf_counter()
        if "auto" in self.heads:
            m = self.heads["auto"]
            p = m.probs(cut(img, m.head))[0]
            o.auto = m.classes[int(p.argmax())]
            o.auto_p_off = float(p[m.classes.index("off")])
            o.auto_probs = p
        if "enemies" in self.heads:
            m = self.heads["enemies"]
            c = cut(img, m.head)
            o.enemies = m.classes[int(m.probs(c)[0].argmax())] == "shown"
            if o.enemies and self.digits is not None:
                o.enemy_count = self.digits.read(c).value
        if "gf_banner" in self.heads:
            m = self.heads["gf_banner"]
            o.gf_banner = m.classes[int(m.probs(cut(img, m.head))[0].argmax())] == "shown"
        t2 = time.perf_counter()
        if self.menu_clock is not None and o.screen in (None, "modes_menu"):
            band = cut_field(img, FIELDS["menu_clock"])
            if on_dark_bar(band, FIELDS["menu_clock"]):
                o.menu_clock = self.menu_clock.read_crop(band).value
        t3 = time.perf_counter()
        o.ms = {"screen": 1000 * (t1 - t0), "crops": 1000 * (t2 - t1), "text": 1000 * (t3 - t2)}
        return o
