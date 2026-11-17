"""Checks on the user's real frames (skipped without them). Nothing from the frames is stored in the repo."""

import random

import cv2
import numpy as np
import pytest

from adbot2 import paths
from adbot2.dataset import load_game_area, load_manifest
from adbot2.geometry import VALID_BOTTOM, VALID_TOP

pytestmark = pytest.mark.data


def sample(layout_dpi, n, seed=0):
    recs = [r for r in load_manifest(paths.manifest_path()).values()
            if r.root == "captures" and r.full_frame and r.dpi == layout_dpi and r.file.endswith(".jpg")]
    random.Random(seed).shuffle(recs)
    return recs[:n]


def best_offset(img, tmpl, x, y, pad):
    reg = img[y - pad:y + tmpl.shape[0] + pad, x - pad:x + tmpl.shape[1] + pad]
    res = cv2.matchTemplate(reg, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return score, (loc[0] - pad, loc[1] - pad)


def test_every_layout_maps_and_outside_stays_blank():
    for dpi in (96, 144):
        for r in sample(dpi, 10):
            a = load_game_area(r, paths.home()).image
            assert not a[:VALID_TOP].any() and not a[VALID_BOTTOM:].any()
            assert a[VALID_TOP:VALID_BOTTOM].mean() > 5  # the game band is filled


def test_bottom_hud_lines_up_across_layouts():
    """The skill hotbar (bottom-anchored) sits at the same canonical spot at 100% and 150% (within 1 px).

    The hotbar is not on every screen, so a few 100% frames are tried as the reference and the best one is kept.
    """
    x, y, w, h = 700, 1000, 300, 38
    targets = [load_game_area(r, paths.home()).image for r in sample(144, 60, seed=2)]
    best: list = []
    for r in sample(96, 8, seed=1):
        ref = load_game_area(r, paths.home()).image[y:y + h, x:x + w].copy()
        if ref.std() < 20:
            continue
        offs = [off for score, off in (best_offset(t, ref, x, y, 30) for t in targets) if score > 0.8]
        if len(offs) > len(best):
            best = offs
    assert len(best) >= 10
    assert all(abs(dx) <= 1 and abs(dy) <= 1 for dx, dy in best), best
