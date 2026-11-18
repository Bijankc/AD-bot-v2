"""Decisions over time instead of single frames.

- ScreenFilter: a forward filter (hidden Markov model) over the screen classes. The belief carries over between
  frames: the screen is assumed to stay the same, with the chance of a change growing with the time since the last
  frame (half of it after `half_life_s`). One odd frame (a white flash read as "loading") then barely moves the
  belief, while a real change wins after a frame or two. If no screen is convincing the answer is None ("unknown"):
  the bot waits and saves the frame for labelling instead of acting on a guess.
- Agree: a value (enemy count, raid clock offset, AUTO state) is taken only after n readings in a row agree.
"""

from __future__ import annotations

import math
from typing import Hashable

import numpy as np


class ScreenFilter:
    def __init__(self, classes: list[str], half_life_s: float = 3.0, min_belief: float = 0.7,
                 floor: float = 1e-3):
        self.classes = list(classes)
        self.k = len(classes)
        self.half_life_s = half_life_s
        self.min_belief = min_belief
        self.floor = floor                      # no class is ever ruled out completely
        self.belief = np.full(self.k, 1.0 / self.k)
        self.t: float | None = None

    def reset(self) -> None:
        self.belief[:] = 1.0 / self.k
        self.t = None

    def update(self, probs: np.ndarray, t: float) -> tuple[str | None, float]:
        """Feed one frame's classifier probabilities at time t (seconds). Returns (screen or None, belief)."""
        p = np.asarray(probs, np.float64)
        p = p / p.sum()
        if self.t is None:
            prior = np.full(self.k, 1.0 / self.k)
        else:
            dt = max(0.0, t - self.t)
            change = 1.0 - math.pow(0.5, dt / self.half_life_s)
            prior = (1.0 - change) * self.belief + change / self.k
        self.t = t
        post = prior * (p + self.floor)
        self.belief = post / post.sum()
        k = int(self.belief.argmax())
        b = float(self.belief[k])
        return (self.classes[k] if b >= self.min_belief else None), b


class Agree:
    """Accept a reading once `n` readings in a row agree (None readings break the run)."""

    def __init__(self, n: int = 2):
        self.n = n
        self.value: Hashable | None = None     # the accepted value
        self._last: Hashable | None = None
        self._run = 0

    def update(self, reading: Hashable | None) -> Hashable | None:
        if reading is None:
            self._last, self._run = None, 0
            return self.value
        if reading == self._last:
            self._run += 1
        else:
            self._last, self._run = reading, 1
        if self._run >= self.n:
            self.value = reading
        return self.value
