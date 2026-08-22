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


# Used by stain_detector.py, spot_detector.py, and find_hole_or_tear_regions
# below. All four defects (holes, tears, stains, spots) are, at the pixel
# level, the same question: "does this patch of the glove look like a
# different material/color than its own immediate surroundings?" - a glove
# is a curved, glossy surface, so comparing against one single "typical
# color" for the whole glove would flag ordinary shading/highlights as
# defects everywhere. Comparing each pixel to a wide LOCAL blur of color
# around it cancels out that slow shading gradient and leaves only real,
# local anomalies. Each caller then decides its own size/shape cutoffs for
# what counts as a hole vs. a tear vs. a stain vs. a spot.
def color_residual_outlier_mask(mask, lab, mad_multiplier, blur_sigma):
    mask_f = (mask == 255).astype(np.float32)
    if not np.any(mask_f):
        return np.zeros(mask.shape, dtype=np.uint8)

    # Blur restricted to glove pixels only (a normal blur would pull in
    # background color near the edges): blur(image * mask) / blur(mask).
    local_avg = cv2.GaussianBlur(lab * mask_f[:, :, None], (0, 0), blur_sigma)
    coverage = cv2.GaussianBlur(mask_f, (0, 0), blur_sigma)
    local_avg /= np.maximum(coverage[:, :, None], 1e-3)

    # Only the A/B (color) channels of the residual, not L (brightness) -
    # this is what tells a real color anomaly apart from a shadow, highlight
    # or wrinkle (same color, different brightness).
    color_residual = np.linalg.norm((lab - local_avg)[:, :, 1:3], axis=2)

    residual_in_mask = color_residual[mask == 255]
    median = np.median(residual_in_mask)
    # "median absolute deviation": how much color naturally varies within
    # the glove itself. max(...,1) avoids divide-by-zero.
    mad = np.maximum(np.median(np.abs(residual_in_mask - median)), 1.0)
    threshold = median + mad_multiplier * mad

    outlier = np.zeros(mask.shape, dtype=np.uint8)
    outlier[(color_residual > threshold) & (mask == 255)] = 255
    return outlier


# Used by hole_detector.py and tear_detector.py, in addition to
# find_internal_regions above.
#
# find_internal_regions only catches a hole/tear when whatever is showing
# through it (skin, a table, background) happens to be close enough to the
# background color sampled by glove_detector.py's border-based segmentation.
# That is not always true - skin tone, for example, is often a color that is
# clearly NOT the glove, but also not particularly close to the sampled
# background either, so it never breaks the coarse foreground mask and
# never shows up as an "island" for find_internal_regions to find.
#
# So this adds a second, independent signal: a big, strong local color
# anomaly inside the glove's own silhouette (the same signal
# stain_detector.py/spot_detector.py use for small marks, via
# color_residual_outlier_mask above), kept only above min_area_ratio so it
# only catches damage-sized patches, not small stains/spots. The two
# signals are drawn into one mask and re-extracted together so a region
# caught by both is not counted twice.
LARGE_ANOMALY_AREA_RATIO = 0.005  # boundary vs. stains/spots - see stain_detector.py's own upper bound


def find_hole_or_tear_regions(glove_result, preprocessed, min_area_ratio=LARGE_ANOMALY_AREA_RATIO,
                               mad_multiplier=6.0, blur_sigma=25):
    raw_mask = glove_result["raw_mask"]
    combined = np.zeros(raw_mask.shape, dtype=np.uint8)
    cv2.drawContours(combined, find_internal_regions(raw_mask), -1, 255, thickness=cv2.FILLED)

    # Eroded much further in from the silhouette than stain_detector.py's
    # own erosion (iterations=2): the glove's TRUE edge - especially the
    # open cuff, where the material gradually blends into wrist/arm skin
    # over a fairly wide band in these photos - already looks like a giant
    # "color anomaly" on its own. A shallow erosion leaves enough of that
    # blend band inside the mask to trip this large-anomaly check by
    # itself, on every photo, regardless of whether real damage is present.
    # A real hole/tear sits well inside the glove, far from that blend
    # band, so this deeper erosion still finds it fine (find_internal_
    # regions above has no such margin and remains the more sensitive of
    # the two candidate sources for damage close to the edge).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(glove_result["mask"], kernel, iterations=6)
    if cv2.countNonZero(inner_mask) > 0:
        lab = preprocessed["lab"].astype(np.float32)
        outlier = color_residual_outlier_mask(inner_mask, lab, mad_multiplier, blur_sigma)
        outlier = cv2.morphologyEx(outlier, cv2.MORPH_CLOSE, kernel)

        glove_area = cv2.contourArea(glove_result["contour"])
        min_area = min_area_ratio * glove_area
        anomaly_contours, _ = cv2.findContours(outlier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_anomalies = [c for c in anomaly_contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(combined, big_anomalies, -1, 255, thickness=cv2.FILLED)

    if cv2.countNonZero(combined) == 0:
        return []
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Used by finger_detector.py and edge_tear_detector.py. Both look for places
# where the glove's real outline dips inward from its convex hull (the
# "rubber band" shape with no dents) - a finger gap is one such dip, and so
# is damage sitting right on the glove's outer edge. They differ only in
# which dips they keep and why, so the shared mechanics (finding the dips,
# measuring each one) live here.
def _angle_degrees(start, end, far):
    # Angle at `far`, between the lines far->start and far->end.
    a = start.astype(np.float32) - far.astype(np.float32)
    b = end.astype(np.float32) - far.astype(np.float32)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def convexity_dips(contour, depth_threshold):
    """Every place the glove's outline dips inward from its convex hull,
    deep enough to matter (>= depth_threshold). Returns a list of dicts
    with "depth", "angle" (degrees, at the dip's deepest point), and the
    "far"/"start"/"end" contour points (full-image coordinates - same
    coordinate space as glove_result["contour"])."""
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return []
    hull_indices = np.sort(hull_indices, axis=0)

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return []

    dips = []
    for start_idx, end_idx, far_idx, depth_fx in defects.reshape(-1, 4):
        depth = depth_fx / 256.0
        if depth < depth_threshold:
            continue
        start = contour[start_idx][0]
        end = contour[end_idx][0]
        far = contour[far_idx][0]
        dips.append({
            "depth": depth,
            "angle": _angle_degrees(start, end, far),
            "far": far,
            "start": start,
            "end": end,
        })
    return dips


def contour_centroid(contour, fallback):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return fallback
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


# How far a hull point must sit from the glove's own centroid, as a
# fraction of the glove's bbox diagonal, to count as "fingertip-like". See
# in_finger_region() below.
MIN_FINGERTIP_DIST_RATIO = 0.28


def in_finger_region(dip, centroid, scale):
    """A dip belongs to the normal finger-gap pattern only if BOTH hull
    points on either side of it (start/end) are themselves fingertip-like:
    far from the glove's own centroid, since a finger is a narrow shape
    that projects a long way out from the palm. Real finger valleys sit
    between two such points easily. Damage elsewhere on the glove (e.g. a
    tear near the wrist) can still be deep and narrow enough to pass a
    depth/angle check alone, but its neighboring hull points sit much
    closer to the centroid - still near the main body of the glove, not
    out at a fingertip - so this catches the difference
    finger_detector.py and edge_tear_detector.py each need, in opposite
    directions.

    Distance from the centroid, not a fixed "up/down" direction, is used
    deliberately: gloves in test photos are not always framed with fingers
    pointing straight up (e.g. an arm/wrist entering the frame from an
    angle), so a fingertip has to be identified by how far it reaches from
    the glove's own center, not by which side of the photo it is on."""
    cx, cy = centroid
    start, end = dip["start"], dip["end"]
    dist_start = np.hypot(start[0] - cx, start[1] - cy) / scale
    dist_end = np.hypot(end[0] - cx, end[1] - cy) / scale
    return min(dist_start, dist_end) >= MIN_FINGERTIP_DIST_RATIO


# A dip this much shallower than the deepest finger-shaped dip in the same
# photo is treated as damage, not a real finger gap - see classify_dips().
SHALLOW_DEPTH_RATIO = 0.3


def classify_dips(dips, centroid, scale, max_finger_angle=120):
    """Splits convexity-defect dips (see convexity_dips() above) into two
    groups, shared by finger_detector.py and edge_tear_detector.py so the
    two stay perfectly consistent - every dip lands in exactly one of them:

      - finger dips: dips that plausibly sit between two real fingers.
        Narrow (angle <= max_finger_angle), with both neighboring hull
        points out at a fingertip (in_finger_region) - AND deep enough
        relative to the deepest such dip found in this same photo. That
        last check matters because a tear or hole can occasionally produce
        a dip that is narrow and has fingertip-distance neighbors just
        from where it happens to sit on an already-elongated part of the
        glove (e.g. near the wrist, on a hand shape with little of the
        wrist cropped out) - but even then, it is usually noticeably
        shallower than the real finger gaps in that same photo, since real
        fingers are longer than any accidental damage-induced notch.
      - other dips: everything else - candidates for edge_tear_detector.py.
    """
    candidates = [d for d in dips if d["angle"] <= max_finger_angle and in_finger_region(d, centroid, scale)]
    deepest = max((d["depth"] for d in candidates), default=0.0)
    cutoff = SHALLOW_DEPTH_RATIO * deepest

    finger_dips = [d for d in candidates if d["depth"] >= cutoff]
    finger_ids = {id(d) for d in finger_dips}
    other_dips = [d for d in dips if id(d) not in finger_ids]
    return finger_dips, other_dips
