import cv2
import numpy as np

from .helpers import (
    classify_dips,
    contour_centroid,
    contour_features,
    convexity_dips,
    draw_contours,
    find_hole_or_tear_regions,
)

# A tear can show up on a glove photo in two different ways at the pixel
# level. This file catches both and reports them as ONE "tears" result (one
# count, one color, one annotated image) - a tear is a single defect type,
# regardless of which of the two ways it happened to appear in a given
# photo:
#
#   1. ENCLOSED tear: a rip that exposes skin/background but is still
#      surrounded by glove material on every side (e.g. a rip in the middle
#      of the palm). This shows up as a "hole" in the topological sense -
#      find_hole_or_tear_regions() in helpers.py finds these (shared with
#      hole_detector.py, which claims the round ones; a tear is what's left
#      over: low circularity, elongated/irregular).
#
#   2. EDGE tear: a rip right at the glove's own outer boundary. It is not
#      "surrounded" by glove material on all sides any more, so signal 1
#      above structurally cannot find it - there is no enclosed island to
#      find, the tear IS the boundary at that point. We find these instead
#      by comparing the glove's outline to its convex hull (the same
#      convexity-defect idea finger_detector.py uses for finger gaps, via
#      convexity_dips()/classify_dips() in helpers.py): a real edge tear is
#      a dip in the outline that classify_dips() does NOT recognize as a
#      normal finger gap, and that is narrow enough to be a cut rather than
#      a natural curve of the glove's silhouette (e.g. where the thumb
#      meets the wrist).
#
# The two signals look at different evidence (enclosed color patches vs.
# outline shape) and structurally can't both fire on the exact same rip, so
# double-counting isn't expected - but _edge_point_is_new() below guards
# against it anyway, in case an edge tear happens to sit right next to an
# enclosed one already found.

TEAR_COLOR = (0, 140, 255)

# --- enclosed tears (signal 1) ---
ENCLOSED_MIN_AREA_PX = 100  # small noise flecks (JPEG chroma noise) are common in dark shadow areas
ENCLOSED_MAX_AREA_RATIO = 0.20
ENCLOSED_MAX_CIRCULARITY = 0.35  # rounder than this - hole_detector.py claims it instead

# --- edge tears (signal 2) ---
EDGE_MIN_DEPTH_RATIO = 0.02  # ignore dips shallower than this vs. the glove's own size (just noise)
EDGE_MAX_ANGLE_DEG = 130     # a natural curve (thumb/pinky into wrist) is close to flat; a cut is narrower
EDGE_DEDUPE_DISTANCE_PX = 25  # how close an edge dip must be to an enclosed tear to count as the same one


def detect_tears(glove_result, preprocessed):
    enclosed_regions, enclosed_metrics = _find_enclosed_tears(glove_result, preprocessed)
    edge_points, edge_metrics = _find_edge_tears(glove_result, enclosed_regions)

    annotated = _draw_tears(glove_result, enclosed_regions, edge_points)
    total_count = len(enclosed_regions) + len(edge_points)

    return {
        "defect": "tears",
        "detected": total_count > 0,
        "count": total_count,
        "regions": enclosed_regions + edge_points,
        "metrics": enclosed_metrics + edge_metrics,
        "annotated_image": annotated,
    }


def _find_enclosed_tears(glove_result, preprocessed):
    glove_area = cv2.contourArea(glove_result["contour"])
    candidates = find_hole_or_tear_regions(glove_result, preprocessed)

    regions, metrics = [], []
    for contour in candidates:
        feats = contour_features(contour)
        if feats["area"] < ENCLOSED_MIN_AREA_PX:
            continue  # too small, probably just noise
        if feats["area"] > ENCLOSED_MAX_AREA_RATIO * glove_area:
            continue  # too big to be a single tear
        if feats["circularity"] >= ENCLOSED_MAX_CIRCULARITY:
            continue  # round enough to be a hole, so hole_detector.py should claim it
        regions.append(contour)
        metrics.append(feats)
    return regions, metrics


def _find_edge_tears(glove_result, enclosed_regions):
    contour = glove_result["contour"]
    x, y, w, h = glove_result["bbox"]
    scale = float(np.hypot(w, h))
    depth_threshold = EDGE_MIN_DEPTH_RATIO * scale
    centroid = contour_centroid(contour, fallback=(x + w / 2.0, y + h / 2.0))

    dips = convexity_dips(contour, depth_threshold)
    _, other_dips = classify_dips(dips, centroid, scale)

    points, metrics = [], []
    for dip in other_dips:
        if dip["angle"] >= EDGE_MAX_ANGLE_DEG:
            continue  # too wide/flat to be a cut - a natural curve of the glove's outline
        point = tuple(dip["far"])
        if _edge_point_is_new(point, enclosed_regions):
            points.append(point)
            metrics.append({"depth": dip["depth"], "angle": dip["angle"]})
    return points, metrics


def _edge_point_is_new(point, enclosed_regions):
    px, py = float(point[0]), float(point[1])
    for contour in enclosed_regions:
        distance = cv2.pointPolygonTest(contour, (px, py), True)
        if distance >= -EDGE_DEDUPE_DISTANCE_PX:
            return False  # inside, or just outside, an already-found enclosed tear
    return True


def _draw_tears(glove_result, enclosed_regions, edge_points):
    x, y, w, h = glove_result["bbox"]
    cropped = glove_result["cropped_bgr"]

    local_contours = [c - [x, y] for c in enclosed_regions]
    annotated = draw_contours(cropped, local_contours, color=TEAR_COLOR)

    for px, py in edge_points:
        cv2.circle(annotated, (px - x, py - y), 10, TEAR_COLOR, 2)

    if enclosed_regions:
        label_x, label_y, _, _ = cv2.boundingRect(local_contours[0])
    elif edge_points:
        label_x, label_y = edge_points[0][0] - x, edge_points[0][1] - y
    else:
        return annotated

    cv2.putText(annotated, "tear", (label_x, max(label_y - 10, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEAR_COLOR, 1, cv2.LINE_AA)
    return annotated
