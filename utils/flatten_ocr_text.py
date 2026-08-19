import re

import html2text


def flatten_ocr_text(text: str) -> str:
    """
    "Flattens" a given piece of OCR text.
    - Strips HTML
    - Coarsely removes hyphenations
    - Coarsely removes line breaks
    """
    text = html2text.html2text(text)
    text = re.sub(r"-\s+(?=[a-z])", "", text)  # Hyphenations
    text = text.replace("\n", " ")  # Flatten
    return text
