"""
utils/preprocess.py
Image preprocessing for receipt scans before OCR.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def resize_max_dim(img: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    """Scale the longest side to max_dim so small receipt text is legible to OCR."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if abs(scale - 1) > 0.01:
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize with adaptive threshold — handles uneven lighting on receipts well."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply a light blur and Otsu threshold for receipts with even paper lighting."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotation using the minAreaRect of non-zero pixels."""
    coords = np.column_stack(np.where(gray < 255))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Skip correction if angle is negligible or looks like noise
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_receipt(path: str, debug: bool = False) -> np.ndarray:
    """
    Full pipeline: load -> resize -> grayscale -> denoise -> deskew -> threshold.
    Returns a preprocessed image ready for OCR.
    Set debug=True during development to skip thresholding (easyocr often does
    better on a clean grayscale image than a hard binary one).
    """
    img = load_image(path)
    img = resize_max_dim(img)
    gray = to_grayscale(img)
    gray = denoise(gray)
    gray = deskew(gray)
    if debug:
        return gray
    return adaptive_threshold(gray)
