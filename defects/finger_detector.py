import cv2
import numpy as np

EXPECTED_FINGERS = 5
MIN_DEPTH_RATIO = 0.02     # valley depth vs. glove bbox diagonal, to count as a finger gap
MAX_ANGLE_DEG = 120        # finger gaps are narrow notches; wider notches are palm/wrist curves


def detect_missing_fingers(glove_result):
    # Fingers are separated by narrow "valleys" on the glove's outline. Deep,
    # WIDE notches also show up (e.g. where the thumb's base curves into the
    # wrist) but are not finger gaps, so we require both a minimum depth and
    # a narrow angle at the valley point to count it as a real finger gap.
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
        if depth < depth_threshold:
            continue

        start = contour[start_idx][0]
        end = contour[end_idx][0]
        far = contour[far_idx][0]
        if _angle_degrees(start, end, far) > MAX_ANGLE_DEG:
            continue

        valleys.append(tuple(far))

    finger_count = len(valleys) + 1

    for px, py in valleys:
        cv2.circle(annotated, (px - x, py - y), 6, (255, 140, 0), 2)
    cv2.putText(annotated, f"fingers: {finger_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2, cv2.LINE_AA)

    return _result(finger_count, valleys, annotated)


def _angle_degrees(start, end, far):
    # Angle at `far`, between the lines far->start and far->end.
    a = start.astype(np.float32) - far.astype(np.float32)
    b = end.astype(np.float32) - far.astype(np.float32)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


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
