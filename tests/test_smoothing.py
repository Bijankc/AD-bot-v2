import numpy as np

from adbot2.smoothing import Agree, ScreenFilter

C = ["fight", "loading", "lobby"]


def onehot(name: str, p: float = 0.9) -> np.ndarray:
    v = np.full(len(C), (1 - p) / (len(C) - 1))
    v[C.index(name)] = p
    return v


def test_one_odd_frame_does_not_flip_the_screen():
    f = ScreenFilter(C)
    for k in range(10):
        f.update(onehot("fight"), k * 0.5)
    screen, _ = f.update(onehot("loading", 0.95), 5.0)     # a white flash read as loading
    assert screen in ("fight", None)                       # never "loading": at most "unknown, wait a frame"
    assert f.update(onehot("fight"), 5.5)[0] == "fight"


def test_a_real_change_wins_within_two_frames():
    f = ScreenFilter(C)
    for k in range(10):
        f.update(onehot("fight"), k * 0.5)
    f.update(onehot("lobby"), 5.0)
    assert f.update(onehot("lobby"), 5.5)[0] == "lobby"


def test_after_a_long_gap_one_frame_is_enough():
    f = ScreenFilter(C)
    for k in range(10):
        f.update(onehot("fight"), k * 0.5)
    assert f.update(onehot("lobby"), 60.0)[0] == "lobby"


def test_unconvincing_frames_give_unknown():
    f = ScreenFilter(C)
    screen, belief = f.update(np.array([0.4, 0.35, 0.25]), 0.0)
    assert screen is None and belief < 0.7


def test_agree_needs_n_in_a_row():
    a = Agree(2)
    assert a.update(12) is None
    assert a.update(13) is None          # 12 then 13: no agreement yet
    assert a.update(13) == 13
    assert a.update(1) == 13             # one reading of 1 is not enough
    assert a.update(None) == 13
    assert a.update(1) == 13
    assert a.update(1) == 1
