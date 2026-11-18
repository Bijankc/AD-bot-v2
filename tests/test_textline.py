import cv2
import numpy as np

from adbot2.textline import FIELDS, TextField, TextReader, on_dark_bar, segment

F = TextField("t", (0, 0, 200, 24), max_w=14, pattern=r"(\d\d):(\d\d)$")


def band(text: str, bg: int = 45) -> np.ndarray:
    """Beige text on a dark bar, roughly like the RAID card clock."""
    c = np.full((24, 200, 3), bg, np.uint8)
    cv2.putText(c, text, (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 225, 250), 1, cv2.LINE_AA)
    return c


def reader() -> TextReader:
    gl, ch = [], []
    for s in ("0123456789", "in:"):
        segs = segment(band(s), F)
        texts = list(s) if s != "in:" else ["i", "n", ":"]
        assert len(segs) == len(texts), (s, len(segs))
        gl += [g for _, g in segs]
        ch += texts
    return TextReader.from_examples(F, gl, ch, max_dist=2.5)


def test_reads_a_clock_and_rejects_a_bad_glyph():
    r = reader()
    got = r.read_crop(band("in: 41:28"))
    assert got.value == (41, 28) and got.text.endswith("41:28")
    c = band("in: 41:28")
    c[:, 150:] = 45
    cv2.rectangle(c, (152, 6), (160, 18), (170, 225, 250), -1)  # a blob where the last digits were
    assert r.read_crop(c).value is None  # "41:?" never becomes "4:1x" or similar


def test_colon_dots_are_one_glyph():
    segs = segment(band("1:2"), F)
    assert len(segs) == 3


def test_dark_bar_filter():
    assert on_dark_bar(band("in: 41:28"), F)
    bright = band("in: 41:28", bg=200)
    assert not on_dark_bar(bright, F)


def test_save_load(tmp_path):
    r = reader()
    r.f = FIELDS["menu_clock"]  # saved readers name a known field
    r.save(tmp_path / "r.npz")
    r2 = TextReader.load(tmp_path / "r.npz")
    assert r2.chars == r.chars and r2.f.name == "menu_clock"
