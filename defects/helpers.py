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


# Used by every defect detector that looks for a local color/brightness
# anomaly (holes, tears, stains, spots, knocking): where the glove touches
# the edge of the PHOTO FRAME, that is the wrist/cuff opening - the arm
# just continues off-frame there, it is not the glove's own edge. In these
# test photos the material is loose and rolled at that opening, so it
# blends gradually into wrist/arm skin over a fairly wide band. Without
# this exclusion, every detector above independently rediscovers that
# blend band and misreads it as damage (an exposed-skin "hole", a long
# "tear" running along the cuff, a "stain" where the color shifts from
# glove to skin) - the margin here is the one place that knowledge is
# encoded, so every detector using it stays consistent.
BORDER_TOUCH_PX = 5       # how close to the frame edge counts as "touching"
CUFF_MARGIN_RATIO = 0.25  # how far inward from a touched edge to exclude, vs glove bbox diagonal


def border_touching_margin_mask(glove_result, image_shape):
    """
    255 = safe to run defect analysis on, 0 = excluded margin near a photo
    frame edge the glove's silhouette touches. A photo where the whole
    glove sits comfortably inside the frame (no side touched) is returned
    unchanged (all 255) - this only matters for a cropped-in shot where the
    wrist/arm exits the frame.
    """
    h, w = image_shape[:2]
    x, y, bw, bh = glove_result["bbox"]
    margin = max(1, int(CUFF_MARGIN_RATIO * np.hypot(bw, bh)))

    safe = np.full((h, w), 255, dtype=np.uint8)
    if x <= BORDER_TOUCH_PX:
        safe[:, :margin] = 0
    if y <= BORDER_TOUCH_PX:
        safe[:margin, :] = 0
    if x + bw >= w - BORDER_TOUCH_PX:
        safe[:, w - margin:] = 0
    if y + bh >= h - BORDER_TOUCH_PX:
        safe[h - margin:, :] = 0
    return safe


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
# A pixel's "local average color" is only trustworthy if enough of its own
# blur neighborhood is actually glove (mask==255) - right at any mask
# boundary (the glove's true edge, or a thin finger surrounded by
# background on both sides) too little of the blur kernel lands on real
# glove pixels, so the normalized average there is really an estimate from
# a handful of pixels, not the wide local blur it's supposed to be. That
# makes the residual there unreliable in a way that looks exactly like a
# real anomaly. MIN_RELIABLE_COVERAGE excludes those pixels from ever being
# flagged, instead of trying to erode a fixed margin away from every mask
# boundary - eroding enough to be safe with a wide blur_sigma (see
# LARGE_ANOMALY_BLUR_SIGMA above) would erase whole fingers, which are
# rarely wider than such a margin needs to be.
MIN_RELIABLE_COVERAGE = 0.6


def color_residual_map(mask, lab, blur_sigma):
    """
    For every glove pixel, how far its A/B (color, not brightness) differs
    from the color immediately around it. Returns (residual, median, mad,
    reliable): the full per-pixel residual map, its own median/MAD (median
    absolute deviation) within the glove - computed only from pixels where
    the local average itself is trustworthy (see MIN_RELIABLE_COVERAGE
    above) - and that same reliable/not-reliable mask, which callers use to
    turn this into an adaptive, per-photo outlier threshold (see
    color_residual_outlier_mask below, and find_hole_or_tear_regions'
    "is this shadow or real color?" check further down this file).
    """
    mask_f = (mask == 255).astype(np.float32)
    if not np.any(mask_f):
        zeros = np.zeros(mask.shape, dtype=np.float32)
        return zeros, 0.0, 1.0, np.zeros(mask.shape, dtype=bool)

    # Blur restricted to glove pixels only (a normal blur would pull in
    # background color near the edges): blur(image * mask) / blur(mask).
    coverage = cv2.GaussianBlur(mask_f, (0, 0), blur_sigma)
    local_avg = cv2.GaussianBlur(lab * mask_f[:, :, None], (0, 0), blur_sigma)
    local_avg /= np.maximum(coverage[:, :, None], 1e-3)

    # Only the A/B (color) channels of the residual, not L (brightness) -
    # this is what tells a real color anomaly apart from a shadow, highlight
    # or wrinkle (same color, different brightness).
    residual = np.linalg.norm((lab - local_avg)[:, :, 1:3], axis=2)

    reliable = (mask == 255) & (coverage >= MIN_RELIABLE_COVERAGE)
    residual_in_mask = residual[reliable]
    if residual_in_mask.size == 0:
        return residual, 0.0, 1.0, reliable
    median = np.median(residual_in_mask)
    # "median absolute deviation": how much color naturally varies within
    # the glove itself. max(...,1) avoids divide-by-zero.
    mad = np.maximum(np.median(np.abs(residual_in_mask - median)), 1.0)
    return residual, median, mad, reliable


def color_residual_outlier_mask(mask, lab, mad_multiplier, blur_sigma):
    residual, median, mad, reliable = color_residual_map(mask, lab, blur_sigma)
    threshold = median + mad_multiplier * mad

    outlier = np.zeros(mask.shape, dtype=np.uint8)
    outlier[(residual > threshold) & reliable] = 255
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

# A large hole/tear has an interior color that is fairly uniform (skin, or
# background), so with blur_sigma=25 (the same value stain_detector.py/
# spot_detector.py use for small marks) the local blur near the MIDDLE of a
# wide anomaly ends up averaging mostly the anomaly's OWN pixels, not the
# surrounding glove - so only a thin RING around its true edge clears the
# outlier threshold, and the interior reads as "normal" again. That ring has
# much less area and a much lower circularity than the real hole - enough to
# make hole_detector.py miss it and tear_detector.py misclassify it as a
# tear instead.
#
# Widening blur_sigma itself to fix this (tried first) backfired: a wider
# blur also tracks ordinary smooth lighting gradients across the glove's
# curved surface less tightly, so it started reading normal shading
# elsewhere on a defect-free glove as a false anomaly. So instead we keep
# blur_sigma at the same value that works well for stains/spots, and instead
# CLOSE the thresholded outlier mask with a wide kernel below - that fills
# the hollow middle of a ring-shaped detection back into a solid blob
# without changing what counts as an outlier in the first place. Confirmed
# against a real photo with two ~70-90px wide holes (this recovers their
# real, solid, round shape) and a real defect-free glove (this does not
# introduce a false anomaly the way widening blur_sigma did).
LARGE_ANOMALY_CLOSE_PX = 90

# How far above the glove's own normal color variation a pixel must sit to
# count as "real color evidence" for the shadow-vs-real-damage check below,
# and what fraction of a candidate region needs to clear that bar. A shadow
# (crease, wrinkle) can drag its MEAN residual up somewhat just from noise/
# compression, so checking the mean alone isn't a reliable split - genuine
# exposed skin/background is a strong anomaly across a good chunk of the
# patch, while a shadow is only weakly, patchily elevated. Requiring a
# fraction of clearly-anomalous pixels (not just an elevated average)
# separates the two cleanly - verified against a real photo where deep
# finger-crease shadows, on a dark background, were briefly misread as
# tears by the mean-only version of this check.
SHADOW_REJECTION_MAD_MULTIPLIER = 4.0
SHADOW_REJECTION_MIN_FRACTION = 0.65


def find_hole_or_tear_regions(glove_result, preprocessed, min_area_ratio=LARGE_ANOMALY_AREA_RATIO,
                               mad_multiplier=6.0, blur_sigma=25):
    raw_mask = glove_result["raw_mask"]
    lab = preprocessed["lab"].astype(np.float32)
    combined = np.zeros(raw_mask.shape, dtype=np.uint8)

    # Signal 1: topological islands (find_internal_regions). On its own,
    # this is fooled by a plain SHADOW that happens to be dark enough for
    # glove_detector.py's segmentation to misread as "background peeking
    # through" - e.g. deep creases between fingers, photographed against a
    # dark background, can look just as "not-glove" to a brightness/color-
    # distance test as an actual hole does. A shadow has the same hue as
    # the glove though, just less light on it - only its L (brightness)
    # differs, not its A/B (color) - so we cross-check every topological
    # candidate against real color evidence before trusting it: keep it
    # only if the patch is ALSO a genuine local color anomaly, using the
    # same A/B-only residual signal 2 (below) uses to find damage in the
    # first place.
    #
    # Deliberately NOT intersected with border_touching_margin_mask here:
    # find_internal_regions only ever returns fully enclosed islands (an
    # "internal" RETR_CCOMP contour, surrounded by glove on all sides), so
    # it essentially never rediscovers the cuff blend band on its own (that
    # band is a gradient touching the frame edge, not an enclosed island) -
    # confirmed against these test photos. Excluding a wrist-side margin
    # here as well nearly cost a real, large, centrally-located tear in a
    # portrait-oriented photo where the wrist (and so the margin) reaches
    # much further up the frame than the actual blend band does.
    shallow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shallow_mask = cv2.erode(glove_result["mask"], shallow_kernel, iterations=2)
    residual, median, mad, reliable = color_residual_map(shallow_mask, lab, blur_sigma)
    color_evidence_threshold = median + SHADOW_REJECTION_MAD_MULTIPLIER * mad

    for contour in find_internal_regions(raw_mask):
        region_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        region_residuals = residual[(region_mask == 255) & reliable]
        if region_residuals.size == 0:
            continue  # entirely outside the safety margin from the edge - too close to judge
        anomalous_fraction = np.mean(region_residuals > color_evidence_threshold)
        if anomalous_fraction < SHADOW_REJECTION_MIN_FRACTION:
            continue  # only patchily/weakly elevated - a shadow, not real damage
        cv2.drawContours(combined, [contour], -1, 255, thickness=cv2.FILLED)

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
    deep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(glove_result["mask"], deep_kernel, iterations=6)
    inner_mask = cv2.bitwise_and(inner_mask, border_touching_margin_mask(glove_result, raw_mask.shape))
    if cv2.countNonZero(inner_mask) > 0:
        outlier = color_residual_outlier_mask(inner_mask, lab, mad_multiplier, blur_sigma)
        # Wide close: fills the hollow middle of a large anomaly's ring-
        # shaped detection back into a solid blob - see LARGE_ANOMALY_CLOSE_PX
        # above for why this is done here instead of via a wider blur.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (LARGE_ANOMALY_CLOSE_PX, LARGE_ANOMALY_CLOSE_PX))
        outlier = cv2.morphologyEx(outlier, cv2.MORPH_CLOSE, close_kernel)

        glove_area = cv2.contourArea(glove_result["contour"])
        min_area = min_area_ratio * glove_area
        anomaly_contours, _ = cv2.findContours(outlier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_anomalies = [c for c in anomaly_contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(combined, big_anomalies, -1, 255, thickness=cv2.FILLED)

    if cv2.countNonZero(combined) == 0:
        return []
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Used by finger_detector.py and tear_detector.py. Both look for places
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
    finger_detector.py and tear_detector.py each need, in opposite
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
    groups, shared by finger_detector.py and tear_detector.py so the
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
      - other dips: everything else - candidates for tear_detector.py.
    """
    candidates = [d for d in dips if d["angle"] <= max_finger_angle and in_finger_region(d, centroid, scale)]
    deepest = max((d["depth"] for d in candidates), default=0.0)
    cutoff = SHALLOW_DEPTH_RATIO * deepest

    finger_dips = [d for d in candidates if d["depth"] >= cutoff]
    finger_ids = {id(d) for d in finger_dips}
    other_dips = [d for d in dips if id(d) not in finger_ids]
    return finger_dips, other_dips
