import sys

import cv2

from preprocessing.image_prep import load_image, preprocess
from detection.glove_detector import detect_glove
from defects.hole_detector import detect_holes
from defects.finger_detector import detect_missing_fingers
from defects.stain_detector import detect_stains

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

    print("holes:", "detected" if holes["detected"] else "not detected", holes["count"])
    print("missing_finger:", "detected" if fingers["detected"] else "not detected",
          f"(counted {fingers['finger_count']} fingers)")
    print("stains:", "detected" if stains["detected"] else "not detected", stains["count"])

    cv2.imwrite("annotated.png", stains["annotated_image"])
