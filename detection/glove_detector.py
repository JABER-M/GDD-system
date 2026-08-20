import cv2
import numpy as np

# Assumes the glove is photographed flat on a plain backdrop that contrasts
# in brightness with it (dark backdrop for pale gloves, light for dark ones).


def _largest_external_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def detect_glove(preprocessed, min_area_ratio=0.05):
    bgr = preprocessed["bgr"]
    gray = preprocessed["gray"]
    h, w = bgr.shape[:2]

    # Otsu picks the brightness threshold automatically from the image itself.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) > 0.5 * mask.size:
        mask = cv2.bitwise_not(mask)  # keep the minority region as the glove

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

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
