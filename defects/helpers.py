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
