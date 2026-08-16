from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DINOv2Config:
    """Local/offline DINOv2 configuration.

    This adapter is migrated from the previous ``patchcores`` project, but the
    input path is now OpenCV BGR ROI arrays so it can share the same ROI crops
    with ProductAlignInspector and later C# preprocessing parity tests.
    """

    model_name: str = "dinov2_vits14"
    image_size: int = 224
    embedding_dim: int = 384
    repo_dir: str = "third_party/dinov2"
    weights_path: str = "weights/dinov2_vits14_pretrain.pth"
    pad_value: int = 255


class DINOv2Adapter:
    """Frozen DINOv2 feature extractor returning normalized patch tokens.

    For ViT-S/14 at 224x224, one ROI produces a 16x16 patch-token grid, i.e.
    256 local 384-D descriptors. These descriptors become the PatchCore-style
    normal memory bank for each fixed ROI.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        config: DINOv2Config | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.config = config or DINOv2Config()
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

        if self.config.image_size <= 0 or self.config.image_size % 14 != 0:
            raise ValueError("DINOv2 ViT-S/14 image_size must be a positive multiple of 14")

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def repo_dir(self) -> Path:
        return self._resolve(self.config.repo_dir)

    @property
    def weights_path(self) -> Path:
        return self._resolve(self.config.weights_path)

    @property
    def patch_grid(self) -> int:
        return self.config.image_size // 14

    def load(self) -> None:
        repo_dir = self.repo_dir
        weights_path = self.weights_path

        if not repo_dir.exists() or not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                f"Local DINOv2 repository not found: {repo_dir}\n"
                "Pass --dino-repo to the existing local facebookresearch/dinov2 checkout."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"DINOv2 weights not found: {weights_path}\n"
                "Pass --dino-weights to dinov2_vits14_pretrain.pth."
            )

        print(f"[DINOv2] device: {self.device}", flush=True)
        print(f"[DINOv2] local repo: {repo_dir}", flush=True)
        print(f"[DINOv2] weights: {weights_path}", flush=True)
        print(f"[DINOv2] input: {self.config.image_size}x{self.config.image_size}", flush=True)

        model = torch.hub.load(
            str(repo_dir),
            self.config.model_name,
            source="local",
            pretrained=True,
            weights=str(weights_path),
        )
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        print("[DINOv2] frozen backbone loaded.", flush=True)

    @staticmethod
    def _pad_to_square(image_bgr: np.ndarray, value: int) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty ROI image")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        h, w = image_bgr.shape[:2]
        side = max(h, w)
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        return cv2.copyMakeBorder(
            image_bgr,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(value, value, value),
        )

    def preprocess_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        image = self._pad_to_square(image_bgr, self.config.pad_value)
        image = cv2.resize(
            image,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))

    def _validate_patch_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise RuntimeError(f"Unexpected DINO patch-token shape: {tuple(tokens.shape)}")
        expected_tokens = self.patch_grid * self.patch_grid
        if int(tokens.shape[1]) != expected_tokens:
            raise RuntimeError(
                f"Unexpected token count {tokens.shape[1]}, expected {expected_tokens} "
                f"for {self.config.image_size}px ViT-S/14 input"
            )
        if int(tokens.shape[2]) != self.config.embedding_dim:
            raise RuntimeError(
                f"Unexpected embedding dim {tokens.shape[2]}, expected {self.config.embedding_dim}"
            )

    @torch.inference_mode()
    def patch_tokens_batch(self, images_bgr: Iterable[np.ndarray]) -> np.ndarray:
        """Return normalized DINOv2 patch tokens as [B, N, D]."""
        if self.model is None:
            raise RuntimeError("DINOv2 is not loaded. Call load() first.")

        tensors = [self.preprocess_bgr(image) for image in images_bgr]
        if not tensors:
            raise ValueError("No ROI images supplied")
        batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.model.forward_features(batch)
        tokens = features["x_norm_patchtokens"]
        self._validate_patch_tokens(tokens)
        tokens = F.normalize(tokens, p=2, dim=-1)
        return tokens.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def patch_tokens(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.patch_tokens_batch([image_bgr])[0]

    @torch.inference_mode()
    def cls_embeddings_batch(self, images_bgr: Iterable[np.ndarray]) -> np.ndarray:
        """Return normalized CLS embeddings, useful for diagnostics only."""
        if self.model is None:
            raise RuntimeError("DINOv2 is not loaded. Call load() first.")
        tensors = [self.preprocess_bgr(image) for image in images_bgr]
        if not tensors:
            raise ValueError("No ROI images supplied")
        batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.model.forward_features(batch)
        embeddings = F.normalize(features["x_norm_clstoken"], p=2, dim=-1)
        return embeddings.detach().cpu().numpy().astype(np.float32)
