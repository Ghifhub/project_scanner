"""
utils/ocr.py
EasyOCR wrapper for reading text off preprocessed receipt images.
"""

from functools import lru_cache
from typing import List, Tuple

import easyocr
import numpy as np

# easyocr result item: (bbox, text, confidence)
OcrResult = Tuple[list, str, float]


@lru_cache(maxsize=1)
def get_reader(languages: Tuple[str, ...] = ("id", "en")) -> easyocr.Reader:
    """
    Cache the Reader so model weights are loaded only once per process
    (loading it per-call is slow — this matters a lot in a streamlit app
    that reruns the script on every interaction).
    """
    return easyocr.Reader(list(languages), gpu=False)


def run_ocr(image: np.ndarray, languages: Tuple[str, ...] = ("id", "en")) -> List[OcrResult]:
    """
    Run OCR on a preprocessed image (grayscale or binarized numpy array).
    Returns a list of (bbox, text, confidence) sorted top-to-bottom, left-to-right.
    """
    reader = get_reader(languages)
    results = reader.readtext(image)

    def sort_key(item: OcrResult):
        bbox = item[0]
        y = min(pt[1] for pt in bbox)
        x = min(pt[0] for pt in bbox)
        return (round(y / 10), x)  # bucket rows so near-same-height text stays on one "line"

    return sorted(results, key=sort_key)


def results_to_lines(results: List[OcrResult], min_confidence: float = 0.3) -> List[str]:
    """
    Collapse OCR results into a list of text lines, grouping words that
    share roughly the same vertical position (same row on the receipt).
    """
    filtered = [r for r in results if r[2] >= min_confidence]
    lines: List[Tuple[float, List[str]]] = []

    for bbox, text, _conf in filtered:
        y = min(pt[1] for pt in bbox)
        placed = False
        for line_y, words in lines:
            if abs(line_y - y) < 15:  # same row tolerance in pixels
                words.append(text)
                placed = True
                break
        if not placed:
            lines.append((y, [text]))

    lines.sort(key=lambda l: l[0])
    return [" ".join(words) for _y, words in lines]


def extract_text_lines(image: np.ndarray, languages: Tuple[str, ...] = ("id", "en")) -> List[str]:
    """Convenience: run OCR and return clean text lines in reading order."""
    results = run_ocr(image, languages)
    return results_to_lines(results)