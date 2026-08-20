import cv2

from .helpers import contour_features, draw_contours

MIN_AREA_PX = 25
MAX_AREA_RATIO = 0.15
MIN_CIRCULARITY = 0.35


def detect_holes(glove_result):
    # A hole = background showing through the glove, i.e. an internal contour
    # (has a parent) in the glove's raw mask, found via RETR_CCOMP hierarchy.
    raw_mask = glove_result["raw_mask"]
    glove_area = cv2.contourArea(glove_result["contour"])
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]

    contours, hierarchy = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    if hierarchy is not None:
        for contour, h_row in zip(contours, hierarchy[0]):
            if h_row[3] == -1:
                continue  # outer glove boundary, not a hole

            feats = contour_features(contour)
            if feats["area"] < MIN_AREA_PX:
                continue
            if feats["area"] > MAX_AREA_RATIO * glove_area:
                continue
            if feats["circularity"] < MIN_CIRCULARITY:
                continue

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
