import cv2
import numpy as np

from .helpers import contour_features, draw_contours

MAD_MULTIPLIER = 3.5   # pixels beyond this many MADs from the median color = outlier
MIN_AREA_PX = 20
MIN_AREA_RATIO = 0.0008


def detect_stains(glove_result, preprocessed):
    # A stain = pixels whose LAB color deviates far from the glove's own
    # median color (uses the image's own stats, so it adapts per photo).
    mask = glove_result["mask"]
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    glove_area = cv2.contourArea(contour)

    lab = preprocessed["lab"]
    glove_pixels = lab[mask == 255].astype(np.float32)
    if glove_pixels.shape[0] == 0:
        return _empty_result(cropped)

    median_color = np.median(glove_pixels, axis=0)
    mad = np.maximum(np.median(np.abs(glove_pixels - median_color), axis=0), 1.0)

    distance = np.linalg.norm((lab.astype(np.float32) - median_color) / mad, axis=2)
    outlier_mask = np.zeros(mask.shape, dtype=np.uint8)
    outlier_mask[(distance > MAD_MULTIPLIER) & (mask == 255)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, kernel)
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(outlier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    min_area = max(MIN_AREA_PX, MIN_AREA_RATIO * glove_area)
    for c in contours:
        feats = contour_features(c)
        if feats["area"] < min_area:
            continue
        regions.append(c)
        metrics.append(feats)

    local_regions = [c - [x, y] for c in regions]
    annotated = draw_contours(cropped, local_regions, color=(255, 0, 255),
                               label="stain" if local_regions else None)

    return {
        "defect": "stains",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }


def _empty_result(cropped):
    return {"defect": "stains", "detected": False, "count": 0,
            "regions": [], "metrics": [], "annotated_image": cropped.copy()}
