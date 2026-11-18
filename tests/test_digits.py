import cv2
import numpy as np

from adbot2.digits import NUM_X0, NUM_Y0, DigitReader, glyphs

RED = (80, 80, 235)  # BGR, like the game's count


def crop_with(text: str, gap: int = 3, dim: float = 1.0) -> np.ndarray:
    """A synthetic 120x120 `enemies` crop: dark diamond area with red digits drawn at the number spot."""
    c = np.full((120, 120, 3), 40, np.uint8)
    x = NUM_X0 + 8
    for ch in text:
        g = np.zeros((40, 30), np.uint8)
        cv2.putText(g, ch, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cols = np.where(g.max(axis=0) > 0)[0]
        g = g[:, cols[0]:cols[-1] + 1]
        h, w = g.shape
        area = c[NUM_Y0 + 4:NUM_Y0 + 4 + h, x:x + w]
        area[g > 0] = (np.array(RED) * dim).astype(np.uint8)
        x += w + gap
    return c


def reader() -> DigitReader:
    gl, d = [], []
    for k in range(10):
        g = glyphs(crop_with(str(k)))
        assert len(g) == 1
        gl.append(g[0])
        d.append(k)
    return DigitReader.fit(gl, d, max_dist=3.0)


def test_reads_numbers():
    r = reader()
    for v in (0, 7, 13, 24, 31):
        assert r.read(crop_with(str(v))).value == v


def test_touching_digits_are_split():
    r = reader()
    c = crop_with("23", gap=-1)  # glyphs overlap: one blob
    assert len(glyphs(c)) == 2 and r.read(c).value == 23


def test_unreadable_gives_none():
    r = reader()
    assert r.read(np.full((120, 120, 3), 40, np.uint8)).value is None      # no digits
    assert r.read(crop_with("8", dim=0.5)).value is None                    # dimmed by a menu: not red enough
    blob = np.full((120, 120, 3), 40, np.uint8)
    blob[NUM_Y0 + 4:NUM_Y0 + 22, NUM_X0 + 10:NUM_X0 + 24] = RED            # a red square is no digit
    assert r.read(blob).value is None


def test_save_load(tmp_path):
    r = reader()
    r.save(tmp_path / "r.npz")
    r2 = DigitReader.load(tmp_path / "r.npz")
    assert r2.digits == r.digits and r2.read(crop_with("19")).value == 19
