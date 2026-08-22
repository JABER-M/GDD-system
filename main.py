import os
import sys

import cv2

from preprocessing.image_prep import load_image, preprocess
from detection.glove_detector import detect_glove

from defects.helpers import draw_contours
from defects.hole_detector import detect_holes
from defects.finger_detector import detect_missing_fingers
from defects.stain_detector import detect_stains
from defects.tear_detector import detect_tears
from defects.spot_detector import detect_spots
from defects.knock_detector import detect_knocks
from defects.edge_tear_detector import detect_edge_tears


def run_all_detectors(image_path):
    """Run the full GDD pipeline on one image: load -> preprocess -> find the
    glove -> run all six defect detectors on it.

    Returns None if no glove was found in the image. Otherwise returns a
    dictionary with two keys:
      "glove"   -> the glove_result dictionary from detect_glove() (mask,
                   contour, bbox, cropped_bgr, ...), so callers can show the
                   isolated glove itself, before any defect is drawn on it.
      "defects" -> a dictionary where each key is a defect name and each
                   value is that detector's result dictionary (with
                   "detected", "count", "annotated_image", etc).

    The GUI calls this same function so it runs the exact same pipeline as
    the command line version below.
    """
    image = load_image(image_path)
    preprocessed = preprocess(image)
    glove = detect_glove(preprocessed)

    if glove is None:
        return None

    defect_results = {
        "holes": detect_holes(glove, preprocessed),
        "missing_finger": detect_missing_fingers(glove),
        "stains": detect_stains(glove, preprocessed),
        "tears": detect_tears(glove, preprocessed),
        "spots": detect_spots(glove, preprocessed),
        "knocking": detect_knocks(glove, preprocessed),
        "edge_tears": detect_edge_tears(glove),
    }
    return {
        "glove": glove,
        "defects": defect_results,
    }


def build_glove_segmentation_image(glove):
    """Build a preview of the glove right after it has been isolated from
    the background: the cropped photo with only the glove's outer boundary
    drawn on it in green, with no defect markings at all. This is meant to
    be shown before the six defect results, so it is clear which region of
    the photo the defect detectors actually looked at."""
    x, y, w, h = glove["bbox"]
    contour_in_crop = glove["contour"] - [x, y]
    return draw_contours(glove["cropped_bgr"], [contour_in_crop], color=(0, 255, 0))


def save_annotated_images(results, output_folder="outputs"):
    """Save one annotated image per defect into output_folder, plus one
    image of the isolated glove itself, using clear file names so all of
    them can be looked at side by side."""
    os.makedirs(output_folder, exist_ok=True)

    glove_image = build_glove_segmentation_image(results["glove"])
    cv2.imwrite(os.path.join(output_folder, "annotated_glove_segmentation.png"), glove_image)

    for defect_name, result in results["defects"].items():
        file_name = f"annotated_{defect_name}.png"
        file_path = os.path.join(output_folder, file_name)
        cv2.imwrite(file_path, result["annotated_image"])


if __name__ == "__main__":
    image_path = sys.argv[1]

    results = run_all_detectors(image_path)

    if results is None:
        print("no_glove_detected")
        sys.exit(1)

    defects = results["defects"]
    holes = defects["holes"]
    fingers = defects["missing_finger"]
    stains = defects["stains"]
    tears = defects["tears"]
    spots = defects["spots"]
    knocks = defects["knocking"]
    edge_tears = defects["edge_tears"]

    print("holes:", "detected" if holes["detected"] else "not detected", holes["count"])
    print("missing_finger:", "detected" if fingers["detected"] else "not detected",
          f"(counted {fingers['finger_count']} fingers)")
    print("stains:", "detected" if stains["detected"] else "not detected", stains["count"])
    print("tears:", "detected" if tears["detected"] else "not detected", tears["count"])
    print("spots:", "detected" if spots["detected"] else "not detected", spots["count"])
    print("knocking:", "detected" if knocks["detected"] else "not detected", knocks["count"])
    print("edge_tears:", "detected" if edge_tears["detected"] else "not detected", edge_tears["count"])

    save_annotated_images(results)
    print("annotated images saved in the 'outputs' folder")
