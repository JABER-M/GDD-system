import cv2

from .helpers import contour_features, draw_contours

# A tear is a rip in the glove material. Like a hole, it shows up as an
# "internal contour" in the glove's raw mask: a region of background color
# that is fully surrounded by glove color. We find these the same way
# hole_detector.py does, using the RETR_CCOMP contour hierarchy.
#
# The difference is shape. A hole from a needle or a small puncture tends to
# be round. A tear tends to be a long, irregular rip. We use circularity to
# tell them apart:
#   circularity = 1.0 means a perfect circle
#   circularity close to 0.0 means a long, thin, irregular shape
# So we only keep internal regions with LOW circularity and call those tears.

MIN_AREA_PX = 25
MAX_AREA_RATIO = 0.20
MAX_CIRCULARITY = 0.35  # shapes rounder than this are treated as holes, not tears


def detect_tears(glove_result):
    raw_mask = glove_result["raw_mask"]
    glove_area = cv2.contourArea(glove_result["contour"])
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]

    contours, hierarchy = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    regions, metrics = [], []
    if hierarchy is not None:
        for contour, h_row in zip(contours, hierarchy[0]):
            if h_row[3] == -1:
                continue  # this is the outer glove boundary, not an internal region

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
