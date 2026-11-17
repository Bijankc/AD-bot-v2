import numpy as np
import pytest

from adbot2.geometry import CLIENT_BOTTOM, VALID_BOTTOM, VALID_TOP, UnknownLayout, layout_of, model_view, to_canvas


def client(h, w=1920):
    """Each client row filled with its own row number (mod 256) in channel 0, so placement is checkable."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (np.arange(h) % 256)[:, None]
    img[:, :, 1] = 200
    return img


def row_of(canvas, y):
    return int(canvas[y, 5, 0])


def test_layouts():
    assert layout_of(1920, 1017, 96) == "100%"
    assert layout_of(1920, 1046, 144) == "150%"
    assert layout_of(1920, 1040, 144) == "150% clipped"
    assert layout_of(1914, 1000, 144) is None and layout_of(1920, 1113, 144) is None


def test_100_percent_client_starts_at_row_23():
    a = to_canvas(client(1017), 96)
    assert a.image.shape == (1080, 1920, 3) and a.layout == "100%"
    assert row_of(a.image, 23) == 0 and row_of(a.image, 1039) == 1016 % 256


def test_150_percent_client_starts_above_the_screen():
    full = to_canvas(client(1046), 144).image
    clipped = to_canvas(client(1046)[6:], 144).image  # what the logger saves: the top 6 rows are off screen
    assert row_of(full, VALID_TOP) == (VALID_TOP + 6) % 256
    assert np.array_equal(full, clipped)


def test_both_layouts_end_at_the_same_row_and_outside_is_blank():
    for img, dpi in ((client(1017), 96), (client(1046), 144)):
        c = to_canvas(img, dpi).image
        assert c[CLIENT_BOTTOM - 1, 0, 1] == 200
        assert not c[:VALID_TOP].any() and not c[VALID_BOTTOM:].any()


def test_screen_grab_is_masked_not_moved():
    grab = np.full((1080, 1920, 3), 77, np.uint8)
    a = to_canvas(grab)
    assert a.layout == "screen" and a.image[500, 500, 0] == 77
    assert not a.image[:VALID_TOP].any() and not a.image[VALID_BOTTOM:].any()
    assert grab[0, 0, 0] == 77  # the input is not changed


def test_unknown_layouts_are_refused():
    with pytest.raises(UnknownLayout):
        to_canvas(client(1000, 1914), 144)
    with pytest.raises(UnknownLayout):
        to_canvas(np.zeros((520, 1320, 3), np.uint8))  # an old-bot crop


def test_model_view_is_the_common_band():
    v = model_view(to_canvas(client(1017), 96))
    assert v.shape == (1017, 1920, 3) and row_of(v, 0) == 0
