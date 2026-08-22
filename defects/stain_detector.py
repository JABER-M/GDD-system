import cv2
import numpy as np

from .helpers import contour_features, draw_contours

MAD_MULTIPLIER = 6.0    # pixels beyond this many MADs from the local color trend = outlier
BLUR_SIGMA = 25         # size of the "local area" used to estimate normal shading/color
MIN_AREA_PX = 20
MIN_AREA_RATIO = 0.0008
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
    mask_f = (mask == 255).astype(np.float32)

    # Blur restricted to glove pixels only (a normal blur would pull in
    # background color near the edges): blur(image * mask) / blur(mask).
    local_avg = cv2.GaussianBlur(lab * mask_f[:, :, None], (0, 0), BLUR_SIGMA)
    coverage = cv2.GaussianBlur(mask_f, (0, 0), BLUR_SIGMA)
    local_avg /= np.maximum(coverage[:, :, None], 1e-3)

    color_residual = np.linalg.norm((lab - local_avg)[:, :, 1:3], axis=2)  # A,B only

    residual_in_glove = color_residual[mask == 255]
    median = np.median(residual_in_glove)
    mad = np.maximum(np.median(np.abs(residual_in_glove - median)), 1.0)
    threshold = median + MAD_MULTIPLIER * mad

    outlier_mask = np.zeros(mask.shape, dtype=np.uint8)
    outlier_mask[(color_residual > threshold) & (mask == 255)] = 255

    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, kernel)
    outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(outlier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    min_area = max(MIN_AREA_PX, MIN_AREA_RATIO * glove_area)
    for c in contours:
        feats = contour_features(c)
        if feats["area"] < min_area:
            continue
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
