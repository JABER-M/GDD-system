import cv2
import numpy as np

from .helpers import classify_dips, contour_centroid, convexity_dips

# hole_detector.py and tear_detector.py both only look for damage that is
# fully SURROUNDED by glove material (an "island" inside the glove's mask).
# If a tear or hole is close to, or touching, the glove's OUTER edge, it is
# not a closed island anymore - it becomes part of the outer boundary
# itself, so those two detectors miss it completely. This detector
# specifically looks for that missed case: damage right at the edge.
#
# It uses the same CONVEXITY DEFECT idea as finger_detector.py: compare the
# glove's real outline to its CONVEX HULL (the "rubber band" shape with no
# dents in it). Every place the outline dips inward from the hull is one
# "dip" (convexity_dips() in helpers.py). classify_dips() (also shared with
# finger_detector.py) sorts those dips into "finger dips" (fit the normal
# finger-gap pattern) and everything else - this file treats every dip in
# that second group as a possible edge tear, as long as it is still narrow
# enough to be a notch rather than a wide, flat, natural curve of the
# outline (e.g. where the thumb or pinky meets the wrist).

MIN_DEPTH_RATIO = 0.02     # ignore dips shallower than this vs the glove's own size (just noise)
MAX_ANGLE_DEG = 130         # a natural curve (thumb/pinky into wrist) is close to flat; a cut is narrower


def detect_edge_tears(glove_result):
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]
    scale = float(np.hypot(w, h))
    depth_threshold = MIN_DEPTH_RATIO * scale
    centroid = contour_centroid(contour, fallback=(x + w / 2.0, y + h / 2.0))

    dips = convexity_dips(contour, depth_threshold)
    _, other_dips = classify_dips(dips, centroid, scale)

    regions, metrics = [], []
    for dip in other_dips:
        if dip["angle"] >= MAX_ANGLE_DEG:
            continue  # too wide/flat to be a cut - a natural curve of the glove's outline
        regions.append(tuple(dip["far"]))
        metrics.append({"depth": dip["depth"], "angle": dip["angle"]})

    local_points = [(px - x, py - y) for (px, py) in regions]
    annotated = cropped.copy()
    for px, py in local_points:
        cv2.circle(annotated, (px, py), 10, (128, 0, 255), 2)
    if local_points:
        px, py = local_points[0]
        cv2.putText(annotated, "edge tear?", (max(px - 40, 5), max(py - 15, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1, cv2.LINE_AA)

    return {
        "defect": "edge_tears",
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "metrics": metrics,
        "annotated_image": annotated,
    }
