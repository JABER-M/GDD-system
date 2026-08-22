import cv2
import numpy as np

from .helpers import classify_dips, contour_centroid, convexity_dips

EXPECTED_FINGERS = 5
MIN_DEPTH_RATIO = 0.02     # valley depth vs. glove bbox diagonal, to count as a finger gap


def detect_missing_fingers(glove_result):
    # Fingers are separated by narrow "valleys" on the glove's outline.
    # classify_dips() (helpers.py, shared with edge_tear_detector.py) picks
    # out the dips that plausibly represent real finger gaps: narrow,
    # sitting between two fingertip-like hull points, and not suspiciously
    # shallower than the other such dips in the same photo (which would
    # suggest damage rather than an actual gap between fingers).
    contour = glove_result["contour"]
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]
    scale = float(np.hypot(w, h))
    depth_threshold = MIN_DEPTH_RATIO * scale
    centroid = contour_centroid(contour, fallback=(x + w / 2.0, y + h / 2.0))

    dips = convexity_dips(contour, depth_threshold)
    finger_dips, _ = classify_dips(dips, centroid, scale)
    valleys = [tuple(dip["far"]) for dip in finger_dips]

    finger_count = len(valleys) + 1

    annotated = cropped.copy()
    for px, py in valleys:
        cv2.circle(annotated, (px - x, py - y), 6, (255, 140, 0), 2)
    cv2.putText(annotated, f"fingers: {finger_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2, cv2.LINE_AA)

    missing = max(0, EXPECTED_FINGERS - finger_count)
    return {
        "defect": "missing_finger",
        "detected": missing > 0,
        "count": missing,
        "finger_count": finger_count,
        "regions": valleys,
        "annotated_image": annotated,
    }
