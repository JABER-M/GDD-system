import cv2
import numpy as np

EXPECTED_FINGERS = 5
MIN_DEPTH_RATIO = 0.08     # valley depth vs. glove bbox diagonal, to count as a finger gap


def detect_missing_fingers(glove_result):
    # Fingers are separated by deep "valleys" on the glove's outline. Count
    # valleys via convexity defects; finger_count = valleys + 1. Fewer
    # fingers than expected = missing/incomplete finger defect.
    contour = glove_result["contour"]
    cropped = glove_result["cropped_bgr"]
    x, y, w, h = glove_result["bbox"]
    depth_threshold = MIN_DEPTH_RATIO * float(np.hypot(w, h))

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    annotated = cropped.copy()

    if hull_indices is None or len(hull_indices) < 3:
        return _result(0, [], annotated)
    hull_indices = np.sort(hull_indices, axis=0)

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return _result(1, [], annotated)

    valleys = []
    for start_idx, end_idx, far_idx, depth_fx in defects.reshape(-1, 4):
        depth = depth_fx / 256.0
        if depth >= depth_threshold:
            valleys.append(tuple(contour[far_idx][0]))

    finger_count = len(valleys) + 1

    for px, py in valleys:
        cv2.circle(annotated, (px - x, py - y), 6, (255, 140, 0), 2)
    cv2.putText(annotated, f"fingers: {finger_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2, cv2.LINE_AA)

    return _result(finger_count, valleys, annotated)


def _result(finger_count, valleys, annotated):
    missing = max(0, EXPECTED_FINGERS - finger_count)
    return {
        "defect": "missing_finger",
        "detected": missing > 0,
        "count": missing,
        "finger_count": finger_count,
        "regions": valleys,
        "annotated_image": annotated,
    }
