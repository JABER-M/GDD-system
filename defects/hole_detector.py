import cv2

from .helpers import contour_features, draw_contours, find_internal_regions

MIN_AREA_PX = 25
MAX_AREA_RATIO = 0.15
MIN_CIRCULARITY = 0.35   # round patch = hole. Elongated patches are left for tear_detector.py.


def detect_holes(glove_result):
    glove_area = cv2.contourArea(glove_result["contour"])
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]

    candidates = find_internal_regions(glove_result["raw_mask"])

    regions, metrics = [], []
    for contour in candidates:
        feats = contour_features(contour)
        if feats["area"] < MIN_AREA_PX:
            continue  # a few stray pixels, not a real hole
        if feats["area"] > MAX_AREA_RATIO * glove_area:
            continue  # too big to plausibly be a hole rather than a bigger tear
        if feats["circularity"] < MIN_CIRCULARITY:
            continue  # not round enough - tear_detector.py handles this shape instead

        regions.append(contour)
        metrics.append(feats)

    local_regions = [c - [x, y] for c in regions]
    annotated = draw_contours(cropped, local_regions, color=(0, 0, 255),
                               label="hole" if local_regions else None)

    return {
        "defect": "holes",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }
