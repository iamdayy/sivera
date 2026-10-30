import io

import cv2
import numpy as np
from PIL import Image, BmpImagePlugin, JpegImagePlugin, PngImagePlugin

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
    # Ultralytics patches Image.open with a plugin auto-installer. Use only our accepted decoders.
    decoder = (JpegImagePlugin.JpegImageFile if content.startswith(b"\xff\xd8\xff") else
               PngImagePlugin.PngImageFile if content.startswith(b"\x89PNG\r\n\x1a\n") else
               BmpImagePlugin.BmpImageFile if content.startswith(b"BM") else None)
    if decoder is None:
        raise ValueError("prediction_invalid_image")
    try:
        with decoder(io.BytesIO(content)) as image:
            if image.width * image.height > config.CAPTURE_MAX_PIXELS:
                raise ValueError("prediction_invalid_image")
            image.verify()
    except (SyntaxError, OSError, Image.DecompressionBombError):
        raise ValueError("prediction_invalid_image") from None
    # Calibration coordinates are raw pixel coordinates, irrespective of EXIF orientation.
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("prediction_invalid_image")
    return image
