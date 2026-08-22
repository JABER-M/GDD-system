import cv2
import numpy as np

# Assumes the glove is photographed flat on a plain background, with the
# glove not touching the frame's outer edges (a border strip of the photo
# has to be pure background so we can sample its color - see _background_color).

BORDER_PX = 15  # how thick a strip around the frame edge counts as "background"


def _background_color(lab):
    # Sample the outer border of the frame as "known background", instead of
    # assuming background is always plain white/dark. This works for ANY
    # background color as long as the glove doesn't touch the frame edges.
    h, w = lab.shape[:2]
    b = min(BORDER_PX, h // 4, w // 4)
    samples = np.concatenate([
        lab[:b, :].reshape(-1, 3), lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ], axis=0)
    median = np.median(samples, axis=0)
    # "median absolute deviation": how much the background color itself
    # varies (lighting falloff, texture). max(...,1) avoids divide-by-zero.
    mad = np.maximum(np.median(np.abs(samples - median), axis=0), 1.0)
    return median, mad


def _foreground_mask(preprocessed):
    lab = preprocessed["lab"].astype(np.float32)
    bg_median, bg_mad = _background_color(lab)

    # Distance of every pixel from the background color, in units of the
    # background's own natural variation. A plain background is close to 0;
    # anything that looks clearly different (the glove) is much higher.
    distance = np.linalg.norm((lab - bg_median) / bg_mad, axis=2)

    # Otsu needs an 8-bit image, so scale distance into 0-255 first. This
    # scaling is data-driven (based on this photo's own distance values), so
    # unlike a fixed distance threshold it adapts to any glove color/material
    # against any background color, as long as they're visually different.
    p99 = np.percentile(distance, 99.5)
    scaled = np.clip(distance / max(p99, 1e-3) * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _largest_external_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def detect_glove(preprocessed, min_area_ratio=0.05):
    bgr = preprocessed["bgr"]
    h, w = bgr.shape[:2]

    mask = _foreground_mask(preprocessed)

    # Remove small speckle noise from the background. We deliberately do NOT
    # also "close" gaps here: doing so tends to partially bridge the narrow
    # gaps between fingers, which pinches off tiny false "holes" right where
    # fingers meet - exactly the kind of artifact hole_detector/tear_detector
    # are looking for, so it creates false positives there.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contour = _largest_external_contour(raw_mask)
    if contour is None or cv2.contourArea(contour) < min_area_ratio * (h * w):
        return None

    filled_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    x, y, bw, bh = cv2.boundingRect(contour)

    return {
        "mask": filled_mask,
        "raw_mask": raw_mask,      # holes still open, used by hole_detector.py
        "contour": contour,
        "bbox": (x, y, bw, bh),
        "cropped_bgr": bgr[y:y + bh, x:x + bw],
    }
