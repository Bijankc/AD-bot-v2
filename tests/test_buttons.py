import cv2
import numpy as np

from adbot2.buttons import START_BOX, read_start


def frame_with(face_bgr, text: str | None) -> np.ndarray:
    """Canonical frame with a button drawn around START_BOX: a horizontal orange gradient or a flat face."""
    img = np.full((1080, 1920, 3), 25, np.uint8)
    x0, y0, x1, y1 = START_BOX
    x0, y0, x1, y1 = x0 - 8, y0 - 6, x1 + 8, y1 + 6
    if face_bgr == "orange":
        for x in range(x0, x1):
            k = (x - x0) / (x1 - x0)          # red-orange (255, 80, 70) to orange (255, 150, 80), as BGR
            img[y0:y1, x] = (70 + 10 * k, 80 + 70 * k, 255)
    else:
        img[y0:y1, x0:x1] = face_bgr
    if text:
        cv2.putText(img, text, (x0 + 30, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return img


def test_orange_start_is_shown():
    r = read_start(frame_with("orange", "Start"))
    assert r.shown, r


def test_dark_back_button_and_plain_screens_are_not():
    assert not read_start(frame_with((40, 34, 34), "Back")).shown
    assert not read_start(np.zeros((1080, 1920, 3), np.uint8)).shown
    assert not read_start(np.full((1080, 1920, 3), 255, np.uint8)).shown      # white flash
    assert not read_start(frame_with("orange", None)).shown                   # orange with no text (fire, a card)
