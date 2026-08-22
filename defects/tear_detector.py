import cv2

from .helpers import contour_features, draw_contours, find_internal_regions

# A tear is a rip in the glove material. Like a hole, it shows up as a
# non-glove patch enclosed within the glove's silhouette (find_internal_
# regions in helpers.py finds and merges these for both hole_detector.py
# and this file). The difference between the two is shape:
#   circularity = 1.0 means a perfect circle (-> a hole, from a puncture)
#   circularity close to 0.0 means a long, irregular rip (-> a tear)
# So we only keep the patches with LOW circularity and call those tears.

MIN_AREA_PX = 25
MAX_AREA_RATIO = 0.20
MAX_CIRCULARITY = 0.35  # shapes rounder than this are treated as holes, not tears


def detect_tears(glove_result):
    glove_area = cv2.contourArea(glove_result["contour"])
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]

    candidates = find_internal_regions(glove_result["raw_mask"])

    regions, metrics = [], []
    for contour in candidates:
        feats = contour_features(contour)
        if feats["area"] < MIN_AREA_PX:
            continue  # too small, probably just noise
        if feats["area"] > MAX_AREA_RATIO * glove_area:
            continue  # too big to be a single tear
        if feats["circularity"] >= MAX_CIRCULARITY:
            continue  # round enough to be a hole, so hole_detector.py should claim it

        regions.append(contour)
        metrics.append(feats)

    local_regions = [c - [x, y] for c in regions]
    annotated = draw_contours(cropped, local_regions, color=(0, 140, 255),
                               label="tear" if local_regions else None)

    return {
        "defect": "tears",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }
