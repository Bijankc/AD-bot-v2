from datetime import datetime, timedelta

import numpy as np

from adbot2.dataset import FrameRecord
from adbot2.evaluation import assign_splits, guarded, regression_set, score, sessions

T0 = datetime(2026, 11, 14, 19, 0)


def rec(i, minutes, dpi):
    t = T0 + timedelta(minutes=minutes)
    return FrameRecord(f"f{i}", "x.jpg", "capture", "x.jpg", 1920, 1017, True, t.isoformat(), root="captures", dpi=dpi)


def test_sessions_are_farm_days_per_layout():
    recs = [rec(0, 0, 96), rec(1, 10, 96), rec(2, 6 * 60, 96), rec(3, 18 * 60, 96), rec(4, 11, 144)]
    s = sessions(recs)
    assert s["f0"] == s["f1"] == s["f2"] == "20261114_100"   # 19:00 to 01:00: one night
    assert s["f3"] == "20261115_100"                         # 13:00 next day: the next farm day
    assert s["f4"] == "20261114_150"                         # other layout = own session


def test_splits_take_turns_per_layout_and_stay_frozen():
    sess = {f"{name}_{j}": name for name in ["20261114_100", "20261116_100", "20261117_100", "20261114_150",
                                              "20261117_150"] for j in range(3)}
    sp = assign_splits(sess, {})
    assert [sp[s] for s in ("20261114_100", "20261116_100", "20261117_100")] == ["train", "test", "val"]
    assert [sp[s] for s in ("20261114_150", "20261117_150")] == ["train", "test"]
    frozen = dict(sp, **{"20261114_100": "val"})
    assert assign_splits(sess, frozen)["20261114_100"] == "val"


def test_guard_band_drops_eval_frames_next_to_train_frames():
    recs = [rec(0, 0, 96), rec(1, 2, 96), rec(2, 30, 96), rec(3, 2, 144)]
    split = {"f0": "train", "f1": "test", "f2": "test", "f3": "test"}
    assert guarded(recs, split) == {"f1"}   # 2 min from a train frame; other layouts do not count


def test_score_counts_none_as_wrong():
    s = score(["a", "a", "b", "b"], ["a", None, "b", "a"], ["a", "b"])
    assert s.accuracy == 0.5 and s.n == 4
    assert s.per_class() == {"a": (0.5, 2), "b": (0.5, 2)}
    assert np.array_equal(s.confusion, [[1, 0], [1, 1]]) and list(s.unanswered) == [1, 0]
    assert "none" in s.table()


def test_regression_set_needs_a_human_label():
    cands = [{"id": "a", "case": "white_flash"}, {"id": "b", "case": "revive"}, {"id": "c", "case": "gf_banner"}]
    reg = regression_set(cands, {"a": "fight_auto", "c": "unsure"})
    assert reg == [{"id": "a", "case": "white_flash", "screen": "fight_auto"}]
