import cv2
import numpy as np

from api.services.image_service import encode_image


def chessboard(x: int = 100, y: int = 90, square: int = 35, blur: float = 0) -> bytes:
    image = np.full((480, 640, 3), 170, np.uint8)
    for row in range(7):
        for col in range(10):
            value = 255 if (row + col) % 2 else 0
            cv2.rectangle(image, (x + col * square, y + row * square),
                          (x + (col + 1) * square - 1, y + (row + 1) * square - 1), (value,) * 3, -1)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return encode_image(image)


def projected_corners() -> tuple[np.ndarray, list[list[list[float]]]]:
    K = np.array([[800., 0, 320], [0, 810., 240], [0, 0, 1]])
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    points = []
    for i in range(12):
        rotation = np.array([0.1 + 0.05 * (i % 4), -0.3 + 0.15 * (i // 4), -0.2 + 0.04 * i])
        translation = np.array([-0.12 + 0.025 * (i % 4), -0.10 + 0.02 * (i // 4), 0.65 + 0.04 * (i % 3)])
        corners, _ = cv2.projectPoints(obj, rotation, translation, K, np.zeros(5))
        points.append(corners.reshape(-1, 2).tolist())
    return K, points
