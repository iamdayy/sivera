import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from api.core.config import Settings, settings


def encode_image(image: np.ndarray, extension: str = ".jpg") -> bytes:
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 85]
    success, encoded = cv2.imencode(extension, image, params)
    if not success:
        raise ValueError("prediction_encoding_failed")
    return encoded.tobytes()


def decode_prediction_image(content: bytes, config: Settings = settings) -> np.ndarray:
    if len(content) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValueError("prediction_image_too_large")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > config.CAPTURE_MAX_PIXELS or image.format not in {"JPEG", "PNG", "BMP"}:
                raise ValueError("prediction_invalid_image")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("prediction_invalid_image") from None
    # Calibration coordinates are raw pixel coordinates, irrespective of EXIF orientation.
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("prediction_invalid_image")
    return image
