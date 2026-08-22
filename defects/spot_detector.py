import cv2
import numpy as np

from .helpers import color_residual_outlier_mask, contour_features, draw_contours

# A spot is a small, roughly round mark on the glove surface (for example a
# drop of dirt or a small blemish) - a local change in the material's actual
# COLOR, not its brightness. That distinction matters: a glove is a curved,
# glossy surface, so brightness alone varies a lot across it even with no
# defect at all (shading, creases, highlights). Comparing every pixel
# against one single "typical color" for the whole glove picks up all of
# that natural shading as false spots.
#
# So instead:
#   1. Estimate the LOCAL "normal" color at every point by blurring the
#      glove with a wide kernel (this follows slow lighting changes but
#      smooths out anything small, like an actual spot).
#   2. Subtract that from the real image: what's left is only small, local
#      anomalies - shading gradients cancel out.
#   3. Only look at the A/B (color) channels of what's left, not L
#      (brightness) - this is what tells a real spot (different color)
#      apart from a shadow/highlight/wrinkle (same color, different
#      brightness).
#   4. Any pixel much farther from the local color than normal (measured in
#      MADs, median absolute deviation) is an outlier. Then keep only the
#      small, round outlier regions - a spot is small and roughly circular;
#      a big or stringy region is more likely a stain or a segmentation
#      edge artifact.
#
# stain_detector.py uses this same local-color-residual idea to find
# stains. A spot is just a stain-like outlier that is small and round, so
# here we reuse the method but keep only the small, round outlier regions.

MAD_MULTIPLIER = 8.0    # how many MADs away from the local color counts as an outlier
BLUR_SIGMA = 25         # size of the "local area" used to estimate normal shading/color
MIN_AREA_PX = 8         # a handful of pixels is more likely noise than a real spot
MAX_AREA_RATIO = 0.003   # spots are small compared to the whole glove
MIN_CIRCULARITY = 0.55   # spots are roughly round; stains are looser/irregular blobs


def detect_spots(glove_result, preprocessed):
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    glove_area = cv2.contourArea(contour)

    # Shrink the mask inward first so pixels right at the glove's silhouette
    # (blended with the background, or sitting in a shading gradient along
    # a curved edge) aren't mistaken for a real spot. Spots look for very
    # small regions, so they need a bigger safety margin from the edge than
    # stain_detector.py does.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(glove_result["mask"], kernel, iterations=4)
    if cv2.countNonZero(mask) == 0:
        return _empty_result(cropped)

    lab = preprocessed["lab"].astype(np.float32)
    outlier_mask = color_residual_outlier_mask(mask, lab, MAD_MULTIPLIER, BLUR_SIGMA)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, small_kernel)

    contours, _ = cv2.findContours(outlier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    max_area = MAX_AREA_RATIO * glove_area
    for c in contours:
        feats = contour_features(c)
        if feats["area"] < MIN_AREA_PX or feats["area"] > max_area:
            continue
        if feats["circularity"] < MIN_CIRCULARITY:
            continue  # not round enough, probably a stain rather than a spot
        regions.append(c)
        metrics.append(feats)

    local_regions = [c - [x, y] for c in regions]
    annotated = draw_contours(cropped, local_regions, color=(0, 255, 255),
                               label="spot" if local_regions else None)

    return {
        "defect": "spots",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }


def _empty_result(cropped):
    return {"defect": "spots", "detected": False, "count": 0,
            "regions": [], "metrics": [], "annotated_image": cropped.copy()}
