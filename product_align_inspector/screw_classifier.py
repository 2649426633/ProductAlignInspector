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
DEFAULT_ARCHITECTURE = "resnet18"
SUPPORTED_ARCHITECTURES = ("resnet18", "mobilenet_v3_small")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESNET18_WEIGHTS = REPO_ROOT / "weights" / "resnet18-f37072fd.pth"


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
    """Build training/validation transforms using deployment-compatible preprocessing."""
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


def _load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found: {path}. "
            "Download the official ResNet18 ImageNet weights to this local path first."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported pretrained weight format: {path}")
    return payload


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained_weights: str | Path | None = None,
) -> nn.Module:
    """Build the ROI state classifier.

    ResNet18 is the production/default architecture. Its ImageNet weights are loaded
    from a local file so training does not depend on internet access. The legacy
    MobileNetV3-Small architecture remains supported for old checkpoints.
    """
    architecture = str(architecture).lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Supported: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        if pretrained:
            weights_path = Path(pretrained_weights) if pretrained_weights else DEFAULT_RESNET18_WEIGHTS
            model.load_state_dict(_load_state_dict_file(weights_path), strict=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # Legacy path kept so existing MobileNetV3 checkpoints/tools keep working.
    if pretrained and pretrained_weights:
        model = models.mobilenet_v3_small(weights=None)
        model.load_state_dict(_load_state_dict_file(pretrained_weights), strict=True)
    else:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def classifier_head(model: nn.Module, architecture: str) -> nn.Module:
    architecture = str(architecture).lower()
    if architecture == "resnet18":
        head = getattr(model, "fc", None)
    elif architecture == "mobilenet_v3_small":
        head = getattr(model, "classifier", None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if head is None:
        raise AttributeError(f"Could not locate classifier head for {architecture}")
    return head


def split_model_parameters(model: nn.Module, architecture: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (backbone_params, head_params) for differential learning rates."""
    head = classifier_head(model, architecture)
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params


def set_backbone_trainable(model: nn.Module, trainable: bool, architecture: str = DEFAULT_ARCHITECTURE) -> None:
    backbone_params, _head_params = split_model_parameters(model, architecture)
    for parameter in backbone_params:
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
    architecture: str = DEFAULT_ARCHITECTURE,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "architecture": str(architecture),
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
    # Keep legacy default for old checkpoints that predate the architecture field.
    architecture = str(checkpoint.get("architecture", "mobilenet_v3_small"))
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        architecture=architecture,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
