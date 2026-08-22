import cv2
import numpy as np


def contour_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "bbox": (x, y, w, h),
    }


def draw_contours(image, contours, color=(0, 0, 255), thickness=2, label=None):
    out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    if label and contours:
        x, y, _, _ = cv2.boundingRect(contours[0])
        cv2.putText(out, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return out


# Used by hole_detector.py and tear_detector.py. Both defects look for the
# same thing at the pixel level - a patch that is NOT glove material inside
# the glove's silhouette (background peeking through a hole, or skin
# showing through a tear) - and only differ in how they classify the SHAPE
# of that patch afterwards (round -> hole, elongated -> tear). So the
# "find the patches" step is shared here; each detector does its own shape
# filtering on the result.
MERGE_KERNEL_PX = 50


def find_internal_regions(raw_mask):
    """
    Find non-glove patches fully enclosed within the glove's silhouette.

    `raw_mask` is the glove mask BEFORE hole-filling (see
    glove_detector.detect_glove): a hole/tear shows up in it as a small
    "island" of background/skin pixels that RETR_CCOMP's contour hierarchy
    reports as an "internal" contour (a contour with a parent).

    On a real, irregular tear this raw search usually returns many small
    disconnected slivers instead of one region - a wrinkle in the fabric,
    or a bit of hair or shadow, breaks what is visually "one tear" into
    several separate pixel blobs. We merge anything within MERGE_KERNEL_PX
    of each other back into single regions before returning them, so the
    result matches what a person would call "one hole" / "one tear", not a
    dozen tiny fragments of it.
    """
    contours, hierarchy = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    internal_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    for contour, h_row in zip(contours, hierarchy[0]):
        if h_row[3] == -1:
            continue  # outer glove boundary, not an internal patch
        cv2.drawContours(internal_mask, [contour], -1, 255, thickness=cv2.FILLED)

    if cv2.countNonZero(internal_mask) == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_KERNEL_PX, MERGE_KERNEL_PX))
    merged_mask = cv2.morphologyEx(internal_mask, cv2.MORPH_CLOSE, kernel)

    merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return merged_contours
