from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def pad_to_square_bgr(image_bgr: np.ndarray, value: int = 255) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    border = (value, value, value) if image_bgr.ndim == 3 else value
    return cv2.copyMakeBorder(
        image_bgr,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border,
    )


def preprocess_bgr_numpy(
    image_bgr: np.ndarray,
    *,
    input_size: int = 224,
    mean: tuple[float, ...] | list[float] = DEFAULT_MEAN,
    std: tuple[float, ...] | list[float] = DEFAULT_STD,
) -> np.ndarray:
    """Canonical deployment preprocessing used by Python, ONNX metadata and C# parity tests.

    Contract:
      BGR uint8 ROI
        -> white pad to square
        -> OpenCV INTER_LINEAR resize
        -> BGR to RGB
        -> float32 / 255
        -> channel-wise mean/std normalization
        -> NCHW float32, batch=1
    """
    image = pad_to_square_bgr(image_bgr, 255)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean_arr) / std_arr
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image[None, ...], dtype=np.float32)


def preprocess_bgr_tensor(
    image_bgr: np.ndarray,
    *,
    input_size: int = 224,
    mean: tuple[float, ...] | list[float] = DEFAULT_MEAN,
    std: tuple[float, ...] | list[float] = DEFAULT_STD,
) -> torch.Tensor:
    return torch.from_numpy(
        preprocess_bgr_numpy(image_bgr, input_size=input_size, mean=mean, std=std)[0]
    )


class DeploymentTensorTransform:
    """PIL-compatible transform that intentionally uses the OpenCV deployment path."""

    def __init__(
        self,
        input_size: int = 224,
        mean: tuple[float, ...] | list[float] = DEFAULT_MEAN,
        std: tuple[float, ...] | list[float] = DEFAULT_STD,
    ) -> None:
        self.input_size = int(input_size)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return preprocess_bgr_tensor(
            bgr,
            input_size=self.input_size,
            mean=self.mean,
            std=self.std,
        )
