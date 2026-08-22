import cv2
import numpy as np

from .helpers import LARGE_ANOMALY_AREA_RATIO, color_residual_outlier_mask, contour_features, draw_contours

MAD_MULTIPLIER = 6.0    # pixels beyond this many MADs from the local color trend = outlier
BLUR_SIGMA = 25         # size of the "local area" used to estimate normal shading/color
MIN_AREA_PX = 20
MIN_AREA_RATIO = 0.0008
# A color anomaly this big or bigger is treated as a hole/tear instead (see
# find_hole_or_tear_regions in helpers.py) - a stain is a surface mark, not
# damage, so it should stay well below "damage" scale.
MAX_AREA_RATIO = LARGE_ANOMALY_AREA_RATIO
MIN_CIRCULARITY = 0.4   # real stains are blob-shaped; thin slivers are segmentation edge noise


def detect_stains(glove_result, preprocessed):
    # A stain is a local CHANGE IN COLOR (chroma), not a change in
    # brightness. A glove is a curved, glossy object, so brightness alone
    # varies a lot across it even with no defect at all (shading, creases,
    # highlights) - comparing against one single "typical color" for the
    # whole glove flags most of that shading as a false stain.
    #
    # So instead we:
    #   1. Estimate the LOCAL "normal" color at every point by blurring the
    #      glove with a wide kernel (this follows slow lighting changes but
    #      smooths out anything small, like an actual stain).
    #   2. Subtract that from the real image: what's left is only the small,
    #      local anomalies - shading gradients cancel out.
    #   3. Only look at the A/B (color) channels of that residual, not L
    #      (brightness) - this is what tells a real stain (different color)
    #      apart from a shadow/highlight/wrinkle (same color, different
    #      brightness).
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    glove_area = cv2.contourArea(contour)

    # Shrink the mask inward first so pixels right at the glove's silhouette
    # (blended with the background) aren't mistaken for real glove color.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(glove_result["mask"], kernel, iterations=2)
    if cv2.countNonZero(mask) == 0:
        return _empty_result(cropped)

    lab = preprocessed["lab"].astype(np.float32)
    outlier_mask = color_residual_outlier_mask(mask, lab, MAD_MULTIPLIER, BLUR_SIGMA)
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, kernel)
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(outlier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    min_area = max(MIN_AREA_PX, MIN_AREA_RATIO * glove_area)
    max_area = MAX_AREA_RATIO * glove_area
    for c in contours:
        feats = contour_features(c)
        if feats["area"] < min_area or feats["area"] >= max_area:
            continue  # too big to be a stain - hole/tear detection handles damage that size
        if feats["circularity"] < MIN_CIRCULARITY:
            continue  # thin sliver, not a blob-shaped stain
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
