import numpy as np

from adbot2.geometry import to_canvas
from adbot2.screen_model import INPUT_H, INPUT_W, to_tensor_array, view_small


def test_view_small_uses_only_the_game_band():
    img = np.full((1017, 1920, 3), 200, np.uint8)        # a 100% capture: the whole client is game
    a = view_small(to_canvas(img, 96))
    assert a.shape == (INPUT_H, INPUT_W, 3) and a.min() == 200
    b = view_small(to_canvas(np.full((1040, 1920, 3), 200, np.uint8), 144))   # 150% clipped: same view
    assert np.array_equal(a, b)


def test_tensor_is_normalised_rgb():
    x = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    x[..., 2] = 255                                       # pure red in BGR
    t = to_tensor_array(x)
    assert t.shape == (1, 3, INPUT_H, INPUT_W) and t.dtype == np.float32
    assert t[0, 0].mean() > 2 and t[0, 2].mean() < -1    # red channel high, blue low


def test_perception_without_models(tmp_path):
    import numpy as np

    from adbot2.perception import Perception

    o = Perception(tmp_path).observe(np.zeros((1080, 1920, 3), np.uint8))  # no models yet: empty fields, no crash
    assert o.screen is None and o.auto is None and o.enemy_count is None and not o.auto_is_off
