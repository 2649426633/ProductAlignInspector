from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


class DinoPatchTokenWrapper(torch.nn.Module):
    """Export only the normalized DINOv2 patch-token tensor.

    Input preprocessing intentionally stays outside ONNX so Python and C# can
    share the exact same OpenCV-compatible BGR -> padded square -> RGB ->
    ImageNet normalization pipeline.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]
        return F.normalize(tokens, p=2, dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description="Export local DINOv2 ViT-S/14 patch tokens to ONNX.")
    p.add_argument("--dino-repo", required=True, help="Local facebookresearch/dinov2 repository")
    p.add_argument("--weights", required=True, help="dinov2_vits14_pretrain.pth")
    p.add_argument("--output", default="artifacts/runtime_bundle/dinov2_vits14_patchtokens.onnx")
    p.add_argument("--device", default="cpu", help="cpu is recommended for export")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    if args.image_size <= 0 or args.image_size % 14 != 0:
        raise SystemExit("--image-size must be a positive multiple of 14")

    repo = Path(args.dino_repo).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 local repository not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv2 weights not found: {weights}")

    device = torch.device(args.device)
    print("=== DINOv2 ONNX Export ===", flush=True)
    print(f"Repo:       {repo}", flush=True)
    print(f"Weights:    {weights}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Image size: {args.image_size}", flush=True)
    print(f"Opset:      {args.opset}", flush=True)

    backbone = torch.hub.load(
        str(repo),
        "dinov2_vits14",
        source="local",
        pretrained=True,
        weights=str(weights),
    )
    backbone.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    wrapper = DinoPatchTokenWrapper(backbone).eval().to(device)
    dummy = torch.zeros((1, 3, args.image_size, args.image_size), dtype=torch.float32, device=device)

    with torch.inference_mode():
        sample = wrapper(dummy)
    expected_tokens = (args.image_size // 14) ** 2
    expected_dim = 384
    expected_shape = (1, expected_tokens, expected_dim)
    if tuple(sample.shape) != expected_shape:
        raise RuntimeError(f"Unexpected patch-token shape {tuple(sample.shape)}, expected {expected_shape}")

    print(f"Output:     {tuple(sample.shape)}", flush=True)
    print("Exporting...", flush=True)

    # Deliberately do not import/use onnxruntime here. The project can export
    # ONNX even on machines whose Python ORT native DLL environment is broken.
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["patch_tokens"],
        dynamic_axes={
            "input": {0: "batch"},
            "patch_tokens": {0: "batch"},
        },
    )

    size_mb = output.stat().st_size / (1024.0 * 1024.0)
    print(f"ONNX:       {output}", flush=True)
    print(f"Size:       {size_mb:.2f} MB", flush=True)
    print("Input:      float32 NCHW [B,3,224,224] AFTER preprocessing", flush=True)
    print(f"Output:     float32 [B,{expected_tokens},{expected_dim}], L2-normalized", flush=True)
    print("ORT check:  skipped intentionally", flush=True)


if __name__ == "__main__":
    main()
