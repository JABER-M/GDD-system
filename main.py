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

if __name__ == "__main__":
    image_path = sys.argv[1]

    image = load_image(image_path)
    preprocessed = preprocess(image)
    glove = detect_glove(preprocessed)

    if glove is None:
        print("no_glove_detected")
        sys.exit(1)

    holes = detect_holes(glove)
    fingers = detect_missing_fingers(glove)
    stains = detect_stains(glove, preprocessed)
    tears = detect_tears(glove)
    spots = detect_spots(glove, preprocessed)
    knocks = detect_knocks(glove, preprocessed)

    print("holes:", "detected" if holes["detected"] else "not detected", holes["count"])
    print("missing_finger:", "detected" if fingers["detected"] else "not detected",
          f"(counted {fingers['finger_count']} fingers)")
    print("stains:", "detected" if stains["detected"] else "not detected", stains["count"])
    print("tears:", "detected" if tears["detected"] else "not detected", tears["count"])
    print("spots:", "detected" if spots["detected"] else "not detected", spots["count"])
    print("knocking:", "detected" if knocks["detected"] else "not detected", knocks["count"])

    cv2.imwrite("annotated.png", knocks["annotated_image"])
