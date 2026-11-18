from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from adbot2.crops import (HEADS, Candidate, CropCache, CropLabels, auto_press_times, cut, model_input, pick_page,
                          propose)
from adbot2.oldlog import LogLine

T0 = datetime(2026, 11, 17, 12, 0, 0)


def rec(rule=None, t=T0):
    return SimpleNamespace(rule=rule, time=t.isoformat())


def test_heads_are_consistent():
    for h in HEADS.values():
        assert len(h.keys) == len(h.classes) == len(set(h.keys))
        x0, y0, x1, y1 = h.box
        assert 0 <= x0 < x1 <= 1920 and 23 <= y0 < y1 <= 1040  # inside the band that is game in every layout


def test_cut_and_model_input_shapes():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[790:910, 680:800] = (0, 0, 255)  # red in BGR
    for h in HEADS.values():
        c = cut(frame, h)
        w, hh = h.store_wh
        assert c.shape == (hh, w, 3)
        x = model_input(np.stack([c, c]), h)
        assert x.shape == (2, 3, h.size[1], h.size[0]) and x.dtype == np.float32
    x = model_input(cut(frame, HEADS["auto"]), HEADS["auto"])
    assert np.allclose(x[0, 0], 1.0) and np.allclose(x[0, 2], 0.0)  # RGB order, [0, 1]


def test_cache_roundtrip(tmp_path):
    cache = CropCache(tmp_path)
    for name, ids in (("a", ["x", "y"]), ("b", ["z"])):
        crops = {k: np.full((len(ids), h.store_wh[1], h.store_wh[0], 3), len(ids), np.uint8) for k, h in HEADS.items()}
        cache.add(name, ids, crops, skipped=["bad"] if name == "b" else [])
    ids, arr = cache.load("auto")
    assert ids == ["x", "y", "z"] and arr.shape[0] == 3 and arr[2, 0, 0, 0] == 1
    assert cache.skipped() == {"bad"}


def test_labels_latest_wins(tmp_path):
    lab = CropLabels(tmp_path / "c.jsonl")
    lab.set_many("auto", {"a": "on", "b": "gone", "c": "nonsense"})
    lab.set_many("auto", {"a": "off"})
    lab.set_many("enemies", {"a": "shown"})
    again = CropLabels(tmp_path / "c.jsonl")
    assert again.of_head("auto") == {"a": "off", "b": "gone"} and again.get("a", "enemies") == "shown"


def test_propose_auto():
    press = auto_press_times([LogLine(T0 + timedelta(seconds=2), "raid: Auto pressed (try 1)"),
                              LogLine(T0, "raid: Auto, waiting for the end")])
    assert press == [T0 + timedelta(seconds=2)]
    assert propose("auto", rec("raid_auto"), "fight", press) == "off"          # 2 s before the press
    assert propose("auto", rec("raid_auto", T0 - timedelta(seconds=5)), "fight", press) == "on"
    assert propose("auto", rec("titan_fight"), "fight", press) == "gone"       # Dimensions have no AUTO
    assert propose("auto", rec(), "lobby") == "gone"
    assert propose("auto", rec(), "dead") == "gone"
    assert propose("auto", rec(), "unsure") is None and propose("auto", rec(), None) is None
    assert propose("enemies", rec(), "respawn") == "shown" and propose("enemies", rec(), "lobby") == "gone"
    assert propose("gf_banner", rec("gf_queued"), "lobby") == "shown"


def test_pick_page_puts_disagreements_first_and_spreads_values():
    cands = [Candidate(f"on{k}", T0 + timedelta(seconds=10 * k), "on", 0.99, "on") for k in range(100)]
    cands += [Candidate(f"off{k}", T0 + timedelta(seconds=10 * k + 5), "off", 0.99, "off") for k in range(3)]
    cands += [Candidate("dis", T0 + timedelta(hours=1), "gone", 0.97, "on"),
              Candidate("low", T0 + timedelta(hours=2), "on", 0.5, "on")]
    page = pick_page(cands, 10)
    ids = [c.id for c in page]
    assert len(page) == 10 and {"dis", "low"} <= set(ids)
    assert sum(i.startswith("off") for i in ids) == 3  # the rare value is shown in full
    times = sorted(c.time for c in page)
    assert all((b - a).total_seconds() >= 4 for a, b in zip(times, times[1:]))


def test_chunked_array_indexes_like_one_array():
    from adbot2.crops import ChunkedArray

    a, b = np.arange(6).reshape(3, 2), np.arange(6, 10).reshape(2, 2)
    c = ChunkedArray([a, b])
    whole = np.concatenate([a, b])
    assert len(c) == 5 and c.shape == (5, 2)
    assert (c[3] == whole[3]).all() and (c[-1] == whole[-1]).all()
    assert (c[[4, 0, 2]] == whole[[4, 0, 2]]).all() and (c[1:4] == whole[1:4]).all()
    assert (np.asarray(c) == whole).all()
