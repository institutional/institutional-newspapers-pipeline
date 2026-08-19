import PIL.Image
import PIL.ImageOps
import numpy as np
import cv2

from const import (
    SCAN_CLAHE_CLIP_LIMIT,
    SCAN_CLAHE_TILE_GRID_SIZE,
    SCAN_AUTOCONTRAST_CUTOFF,
)


def process_scan(image_bytes: bytes) -> PIL.Image.Image:
    """Applies CLAHE and autocontrast to raw image bytes. Returns a processed PIL Image."""
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_array is None:
        raise ValueError("Could not decode image bytes")

    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # CLAHE on the L channel of LAB color space
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=SCAN_CLAHE_CLIP_LIMIT,
        tileGridSize=SCAN_CLAHE_TILE_GRID_SIZE,
    )
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    image_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    pil_image = PIL.Image.fromarray(image_array)

    pil_image = PIL.ImageOps.autocontrast(
        pil_image,
        cutoff=SCAN_AUTOCONTRAST_CUTOFF,
    )

    return pil_image
