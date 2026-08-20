import cv2


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_max_dim(image, max_dim=900):
    h, w = image.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def denoise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=60, sigmaSpace=60)


def correct_illumination(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(image, max_dim=900):
    resized = resize_max_dim(image, max_dim=max_dim)
    denoised = denoise(resized)
    corrected = correct_illumination(denoised)

    return {
        "bgr": corrected,
        "gray": cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY),
        "hsv": cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB),
    }
