"""Exact port of the training preprocessing: crop to fundus + CLAHE + resize.

Used identically for Stage 1 (128x128) and Stage 2 (224x224) inputs.
Keep in sync with the training notebooks - the INT8 models were calibrated
with exactly this pipeline.
"""
import cv2
import numpy as np


def crop_square(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    if mask.sum() > 100:
        ys, xs = np.where(mask)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    y = (s - h) // 2
    x = (s - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def _pipeline(img, size):
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    img = crop_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(path, size):
    """Load an image from disk and run the full pipeline. Returns BGR uint8."""
    return _pipeline(cv2.imread(str(path)), size)


def preprocess_bgr(img, size):
    """Same pipeline for an already-loaded BGR image (used by the demo app)."""
    return _pipeline(img, size)
