import numpy as np

from adbot2.recorder import SavePolicy, change, thumb


def img(v):
    return np.full((1080, 1920, 3), v, np.uint8)


def test_thumb_and_change():
    a, b = thumb(img(10)), thumb(img(40))
    assert a.shape == (36, 64)
    assert change(a, a) == 0
    assert change(a, b) == 30
    assert change(None, a) == 255


def test_first_frame_saved_then_rate_limited():
    p = SavePolicy(change_threshold=6, min_gap=2, max_gap=20)
    a, b = thumb(img(10)), thumb(img(100))
    assert p.should_save(0, a)[0]
    p.saved(0, a, 100)
    assert not p.should_save(1, b)[0]   # changed, but too soon
    assert p.should_save(2, b)[0]       # changed and min_gap passed


def test_static_screen_saved_only_by_max_gap():
    p = SavePolicy(min_gap=2, max_gap=20)
    a = thumb(img(10))
    p.saved(0, a, 1)
    assert not p.should_save(10, a)[0]
    assert p.should_save(20, a)[0]


def test_small_noise_ignored():
    p = SavePolicy(change_threshold=6, min_gap=2)
    a = thumb(img(10))
    p.saved(0, a, 1)
    assert not p.should_save(5, thumb(img(13)))[0]


def test_budget():
    p = SavePolicy(max_bytes=1000, min_free=50)
    assert p.out_of_budget(100) is None
    assert "disk" in p.out_of_budget(10)
    p.saved(0, thumb(img(1)), 1000)
    assert "budget" in p.out_of_budget(100)


def test_near_repeat_of_an_older_save_is_thinned():
    p = SavePolicy(change_threshold=6, min_gap=2, max_gap=20)
    a, b = thumb(img(10)), thumb(img(100))
    p.saved(0, a, 1, "first")
    p.saved(2, b, 1, "change")
    keep, d, kind = p.should_save(4, thumb(img(11)))  # back to (almost) a: changed vs b, repeats a
    assert d >= 6 and not keep and kind == "dup"
    assert p.should_save(4, thumb(img(200)))[2] == "change"


def test_static_gap_doubles_and_resets():
    p = SavePolicy(min_gap=2, max_gap=20, max_static_gap=160)
    a = thumb(img(10))
    p.saved(0, a, 1, "first")
    t = 0
    for expect in (20, 40, 80, 160, 160):
        assert p.gap_now() == expect
        assert not p.should_save(t + expect - 1, a)[0]
        t += expect
        keep, _, kind = p.should_save(t, a)
        assert keep and kind == "periodic"
        p.saved(t, a, 1, kind)
    p.saved(t + 2, thumb(img(100)), 1, "change")
    assert p.gap_now() == 20


def test_png_due():
    p = SavePolicy(png_every=300)
    a = thumb(img(1))
    assert p.png_due(0)
    p.saved(0, a, 1, "first", png=True)
    assert not p.png_due(299)
    assert p.png_due(300)
    assert not SavePolicy(png_every=0).png_due(0)
