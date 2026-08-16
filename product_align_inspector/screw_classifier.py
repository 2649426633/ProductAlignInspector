from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

from product_align_inspector.deploy_preprocess import DeploymentTensorTransform

DEFAULT_CLASSES = ("empty", "screw")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad a PIL image to a square before training-only augmentation."""

    def __init__(self, fill: int | tuple[int, int, int] = 255) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def build_transforms(input_size: int = 224, train: bool = False) -> transforms.Compose:
    """Build training/validation transforms.

    Validation intentionally uses the exact OpenCV deployment preprocessing path.
    Training may add augmentation first, but its final resize/normalization step is
    also the same deployment path. This avoids PIL-vs-OpenCV validation drift.
    """
    ops: list[object] = []

    if train:
        ops.extend(
            [
                PadToSquare(fill=255),
                transforms.RandomAffine(
                    degrees=7.0,
                    translate=(0.035, 0.035),
                    scale=(0.95, 1.05),
                    fill=255,
                ),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.15,
                ),
            ]
        )

    ops.append(
        DeploymentTensorTransform(
            input_size=input_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
    )
    return transforms.Compose(ops)


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    features = getattr(model, "features", None)
    if features is None:
        raise AttributeError("Expected MobileNetV3 model with .features")
    for parameter in features.parameters():
        parameter.requires_grad = trainable


class ProbabilityWrapper(nn.Module):
    """Export wrapper that returns probabilities instead of raw logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(images), dim=1)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    class_names: Sequence[str],
    input_size: int,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "architecture": "mobilenet_v3_small",
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "class_names": list(class_names),
        "input_size": int(input_size),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "metrics": dict(metrics),
        "preprocess_version": "opencv_deploy_v1",
    }
    if extra:
        payload["extra"] = dict(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    class_names = list(checkpoint.get("class_names", DEFAULT_CLASSES))
    architecture = str(checkpoint.get("architecture", "mobilenet_v3_small"))
    if architecture != "mobilenet_v3_small":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
