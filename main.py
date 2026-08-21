import os
import sys

import cv2

from preprocessing.image_prep import load_image, preprocess
from detection.glove_detector import detect_glove

from defects.hole_detector import detect_holes
from defects.finger_detector import detect_missing_fingers
from defects.stain_detector import detect_stains
from defects.tear_detector import detect_tears
from defects.spot_detector import detect_spots
from defects.knock_detector import detect_knocks


def run_all_detectors(image_path):
    """Run the full GDD pipeline on one image: load -> preprocess -> find the
    glove -> run all six defect detectors on it.

    Returns None if no glove was found in the image. Otherwise returns a
    dictionary where each key is a defect name and each value is that
    detector's result dictionary (with "detected", "count",
    "annotated_image", etc).

    The GUI calls this same function so it runs the exact same pipeline as
    the command line version below.
    """
    image = load_image(image_path)
    preprocessed = preprocess(image)
    glove = detect_glove(preprocessed)

    if glove is None:
        return None

    results = {
        "holes": detect_holes(glove),
        "missing_finger": detect_missing_fingers(glove),
        "stains": detect_stains(glove, preprocessed),
        "tears": detect_tears(glove),
        "spots": detect_spots(glove, preprocessed),
        "knocking": detect_knocks(glove, preprocessed),
    }
    return results


def save_annotated_images(results, output_folder="outputs"):
    """Save one annotated image per defect into output_folder, using a clear
    file name for each defect so all six can be looked at side by side."""
    os.makedirs(output_folder, exist_ok=True)
    for defect_name, result in results.items():
        file_name = f"annotated_{defect_name}.png"
        file_path = os.path.join(output_folder, file_name)
        cv2.imwrite(file_path, result["annotated_image"])


if __name__ == "__main__":
    image_path = sys.argv[1]

    results = run_all_detectors(image_path)

    if results is None:
        print("no_glove_detected")
        sys.exit(1)

    holes = results["holes"]
    fingers = results["missing_finger"]
    stains = results["stains"]
    tears = results["tears"]
    spots = results["spots"]
    knocks = results["knocking"]

    print("holes:", "detected" if holes["detected"] else "not detected", holes["count"])
    print("missing_finger:", "detected" if fingers["detected"] else "not detected",
          f"(counted {fingers['finger_count']} fingers)")
    print("stains:", "detected" if stains["detected"] else "not detected", stains["count"])
    print("tears:", "detected" if tears["detected"] else "not detected", tears["count"])
    print("spots:", "detected" if spots["detected"] else "not detected", spots["count"])
    print("knocking:", "detected" if knocks["detected"] else "not detected", knocks["count"])

    save_annotated_images(results)
    print("annotated images saved in the 'outputs' folder")
