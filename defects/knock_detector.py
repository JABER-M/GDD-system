import cv2
import numpy as np

from .helpers import contour_features, draw_contours

# "Knocking" is a small impact or dent on the glove. When the material gets
# locally deformed like this, it often reflects light differently and shows
# up as a small overexposed / glossy patch. So instead of looking at COLOR
# (like stain_detector.py and spot_detector.py do), we look at BRIGHTNESS:
# the V (value/brightness) channel of the HSV color space.
#
# We use the same median + MAD statistical-outlier idea explained in
# spot_detector.py, applied to brightness:
#   1. Find the MEDIAN brightness of all glove pixels (the "typical"
#      brightness for this photo).
#   2. Find the MAD (Median Absolute Deviation): how much brightness
#      normally varies across the glove.
#   3. A pixel that is much BRIGHTER than the median (by many MADs) is
#      flagged as a possible glossy knock mark.
# We only flag pixels that are brighter than normal, not darker, because a
# shadow or a fold in the material is darker, not a knock mark.

MAD_MULTIPLIER = 3.0
MIN_AREA_PX = 10
MAX_AREA_RATIO = 0.01


def detect_knocks(glove_result, preprocessed):
    mask = glove_result["mask"]
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    glove_area = cv2.contourArea(contour)

    hsv = preprocessed["hsv"]
    brightness_channel = hsv[:, :, 2].astype(np.float32)
    glove_brightness = brightness_channel[mask == 255]
    if glove_brightness.size == 0:
        return _empty_result(cropped)

    median_brightness = np.median(glove_brightness)
    # "median absolute deviation": typical amount the brightness varies
    # across the glove. max(..., 1.0) avoids dividing by zero below.
    mad_brightness = max(np.median(np.abs(glove_brightness - median_brightness)), 1.0)

    z_score = (brightness_channel - median_brightness) / mad_brightness
    bright_outlier = np.zeros(mask.shape, dtype=np.uint8)
    # Only pixels ABOVE the median (a glossy highlight), not below (a shadow
    # or a crease in the material).
    bright_outlier[(z_score > MAD_MULTIPLIER) & (mask == 255)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_outlier = cv2.morphologyEx(bright_outlier, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bright_outlier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    max_area = MAX_AREA_RATIO * glove_area
    for c in contours:
        feats = contour_features(c)
        if feats["area"] < MIN_AREA_PX or feats["area"] > max_area:
            continue
        regions.append(c)
        metrics.append(feats)

    local_regions = [c - [x, y] for c in regions]
    annotated = draw_contours(cropped, local_regions, color=(255, 0, 0),
                               label="knock" if local_regions else None)

    return {
        "defect": "knocking",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }


def _empty_result(cropped):
    return {"defect": "knocking", "detected": False, "count": 0,
            "regions": [], "metrics": [], "annotated_image": cropped.copy()}
