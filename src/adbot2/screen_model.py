"""Screen classifier: canonical model view -> one of the Screen classes.

The same preprocessing is used for training, ONNX export and the live bot, so it lives here. The network is
torchvision's MobileNet-V3-large (BSD-3; ImageNet weights) with a new last layer. Weights trained on the user's frames
live under <ADBOT2_HOME>/models/ and are never committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from adbot2.geometry import GameArea, model_view

INPUT_W, INPUT_H = 448, 240          # the 1920x1017 model view, scaled ~0.235 (aspect within 1.2 %)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def view_small(area: GameArea | np.ndarray) -> np.ndarray:
    """BGR uint8 INPUT_H x INPUT_W from a canonical game area (or an already cut model view of 1017 rows)."""
    img = area.image if isinstance(area, GameArea) else area
    v = model_view(img) if img.shape[0] == 1080 else img
    return cv2.resize(v, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def to_tensor_array(small_bgr: np.ndarray) -> np.ndarray:
    """N x H x W x 3 (or H x W x 3) BGR uint8 -> N x 3 x H x W float32, ImageNet-normalised RGB."""
    x = small_bgr[None] if small_bgr.ndim == 3 else small_bgr
    x = x[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def build(n_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
    return net


class OnnxScreenModel:
    """Runs an exported model (model.onnx + classes.json in one folder) with onnxruntime."""

    def __init__(self, folder: Path, gpu: bool = True):
        import onnxruntime as ort

        self.classes: list[str] = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(folder / "model.onnx"), providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    def probs(self, small_bgr: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {self.inp: to_tensor_array(small_bgr)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, area: GameArea | np.ndarray) -> tuple[str, float]:
        p = self.probs(view_small(area))[0]
        return self.classes[int(p.argmax())], float(p.max())
