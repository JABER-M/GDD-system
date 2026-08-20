import cv2
import numpy as np

from .helpers import contour_features, draw_contours

# A spot is a small, roughly round mark on the glove surface (for example a
# drop of dirt or a small blemish). We find it by looking for pixels whose
# color is statistically unusual compared to the rest of the glove.
#
# How the statistics work, in simple terms:
#   1. Convert the image to LAB color space (L = lightness, A/B = color).
#   2. Find the MEDIAN color of all glove pixels. The median is the "typical"
#      glove color for this photo, and is not thrown off by a few odd pixels
#      the way an average can be.
#   3. Find the MAD (Median Absolute Deviation): the median of how far each
#      pixel's color is from that median color. This tells us how much
#      normal color variation to expect (lighting, fabric texture, etc).
#   4. Any pixel much farther from the median than the MAD is an "outlier",
#      i.e. a color that does not belong on this glove.
# This adapts automatically to each photo instead of using one fixed color
# range for every glove.
#
# stain_detector.py uses this exact same idea to find stains. A spot is just
# a stain-like outlier that is small and round, so here we reuse the method
# but keep only the small, round outlier regions.

MAD_MULTIPLIER = 3.5    # how many MADs away from the median counts as an outlier
MIN_AREA_PX = 4
MAX_AREA_RATIO = 0.003   # spots are small compared to the whole glove
MIN_CIRCULARITY = 0.55   # spots are roughly round; stains are looser/irregular blobs


def detect_spots(glove_result, preprocessed):
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
    # "median absolute deviation": typical distance of a glove pixel from the
    # median color. We use max(..., 1.0) so we never divide by zero below.
    mad = np.maximum(np.median(np.abs(glove_pixels - median_color), axis=0), 1.0)

    distance = np.linalg.norm((lab.astype(np.float32) - median_color) / mad, axis=2)
    outlier_mask = np.zeros(mask.shape, dtype=np.uint8)
    outlier_mask[(distance > MAD_MULTIPLIER) & (mask == 255)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, kernel)

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
