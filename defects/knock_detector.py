import cv2
import numpy as np

from .helpers import border_touching_margin_mask, contour_features, draw_contours

# "Knocking" is a small impact or dent on the glove. When the material gets
# locally deformed like this, it often reflects light differently and shows
# up as a small overexposed / glossy patch. So instead of looking at COLOR
# (like stain_detector.py and spot_detector.py do), we look at BRIGHTNESS:
# the V (value/brightness) channel of the HSV color space.
#
# A glove is a curved, glossy surface, so brightness alone already varies a
# lot across it even with no defect at all - the side closer to the light
# is naturally brighter than the side turned away from it. Comparing every
# pixel to one single "typical brightness" for the whole glove would flag
# that entire naturally-brighter side as one giant false knock mark.
#
# So instead of a single global "typical brightness", we compare each pixel
# to the LOCAL brightness around it (a wide blur of the brightness channel).
# That local average follows the slow lighting gradient across the glove,
# but smooths out anything small - like an actual glossy knock mark. What's
# left after subtracting it is only small, local brightness spikes.
#
# We use median + MAD (median absolute deviation) on that local difference,
# the same robust-statistics idea spot_detector.py and stain_detector.py
# use, just applied to brightness instead of color. We only flag pixels
# that are brighter than their surroundings, not darker, because a shadow
# or a fold in the material is darker, not a knock mark.

# Brightness needs a stricter threshold than color does (spot/stain_detector
# use 6.0): a matte, wrinkled glove has a lot of natural micro-highlights
# along its creases, so a looser threshold flags normal texture as knocks.
# 10.0 was tried first and found too strict: a real, deliberately pressed
# knock mark (a firm, rounded press with a smooth object, photographed
# under direct flash) topped out at 9.79x the glove's own natural
# brightness variation - just under the old threshold. Two reference
# photos with no knock at all (one defect-free, one with real holes but no
# knock) topped out at 3.97x and 5.13x, well clear of 8.0, so lowering to
# 8.0 catches real knock marks with a comfortable margin on both sides.
MAD_MULTIPLIER = 8.0
BLUR_SIGMA = 40
MIN_AREA_PX = 10
MAX_AREA_RATIO = 0.01


def detect_knocks(glove_result, preprocessed):
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    glove_area = cv2.contourArea(contour)

    # Shrink the mask inward first so pixels right at the glove's silhouette
    # (blended with the background) aren't mistaken for real glove
    # brightness.
    # Also cut out the wrist/cuff opening margin (see
    # border_touching_margin_mask in helpers.py) - the loose, rolled
    # material there can catch the light differently from the rest of the
    # glove regardless of damage.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(glove_result["mask"], kernel, iterations=2)
    mask = cv2.bitwise_and(mask, border_touching_margin_mask(glove_result, mask.shape))
    if cv2.countNonZero(mask) == 0:
        return _empty_result(cropped)

    brightness = preprocessed["hsv"][:, :, 2].astype(np.float32)
    mask_f = (mask == 255).astype(np.float32)

    # Blur restricted to glove pixels only (a normal blur would pull in
    # background brightness near the edges): blur(v * mask) / blur(mask).
    local_avg = cv2.GaussianBlur(brightness * mask_f, (0, 0), BLUR_SIGMA)
    coverage = cv2.GaussianBlur(mask_f, (0, 0), BLUR_SIGMA)
    local_avg /= np.maximum(coverage, 1e-3)

    brightness_residual = brightness - local_avg

    residual_in_glove = brightness_residual[mask == 255]
    median = np.median(residual_in_glove)
    mad = np.maximum(np.median(np.abs(residual_in_glove - median)), 1.0)
    threshold = median + MAD_MULTIPLIER * mad

    bright_outlier = np.zeros(mask.shape, dtype=np.uint8)
    # Only pixels ABOVE their local surroundings (a glossy highlight), not
    # below (a shadow or a crease in the material).
    bright_outlier[(brightness_residual > threshold) & (mask == 255)] = 255

    bright_outlier = cv2.morphologyEx(bright_outlier, cv2.MORPH_CLOSE, kernel)
    bright_outlier = cv2.morphologyEx(bright_outlier, cv2.MORPH_OPEN, kernel)

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
