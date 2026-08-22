import cv2

# Everything downstream (segmentation, every defect detector) is tuned
# against images produced by preprocess() below, not raw camera photos -
# so this file's job is to make any input photo look as consistent as
# possible before anything else touches it.


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_max_dim(image, max_dim=900):
    # Phone/camera photos come in wildly different resolutions. Every pixel
    # threshold used later (blur kernel sizes, minimum defect area in
    # pixels, etc.) assumes roughly this scale, so without this step the
    # same physical defect would look like a different number of pixels
    # depending on which phone took the photo.
    h, w = image.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return image  # already small enough, don't upscale
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def denoise(image):
    # A bilateral filter blurs flat areas (like the glove's surface) while
    # keeping strong edges (like the glove's outline) sharp - a plain
    # Gaussian blur would soften those edges too, which hurts the
    # contour-based segmentation and defect shape checks used later.
    return cv2.bilateralFilter(image, d=9, sigmaColor=60, sigmaSpace=60)


def correct_illumination(image):
    # CLAHE = "contrast limited adaptive histogram equalization". It
    # boosts local contrast in dim areas of the photo without blowing out
    # already-bright areas (e.g. a camera flash hotspot), which is what a
    # single global brightness/contrast adjustment would do. We only apply
    # it to the L (lightness) channel of LAB, so it evens out lighting
    # without shifting the glove's actual color - important since several
    # defect detectors rely on comparing colors later.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(image, max_dim=900):
    """
    Runs the full preprocessing chain and returns the result in every color
    space the rest of the pipeline needs, so each detector just reads the
    one it wants instead of converting the same image repeatedly:
      - bgr: the cleaned-up photo itself
      - hsv: hue/saturation/VALUE (brightness) - "value" used by knock_detector.py
      - lab: lightness/A/B (color) - used by glove_detector.py and every
             color-based defect detector; more perceptually accurate for
             color comparisons than raw BGR
    """
    resized = resize_max_dim(image, max_dim=max_dim)
    denoised = denoise(resized)
    corrected = correct_illumination(denoised)

    return {
        "bgr": corrected,
        "hsv": cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB),
    }
