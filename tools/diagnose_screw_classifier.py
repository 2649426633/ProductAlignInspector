from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image
from product_align_inspector.screw_classifier import (
    DEFAULT_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    load_checkpoint,
)


def _pad_to_square(image: np.ndarray, value: int = 255) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(value, value, value),
    )


def preprocess_opencv(image_bgr: np.ndarray, input_size: int, mean, std) -> np.ndarray:
    image = _pad_to_square(image_bgr, 255)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean_arr) / std_arr
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image[None, ...], dtype=np.float32)


def _collect_samples(dataset: Path, classes: list[str]) -> list[tuple[Path, str, str]]:
    result: list[tuple[Path, str, str]] = []
    manifest = dataset / "manifest.csv"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok" or row.get("kind") != "screw_slot":
                    continue
                label = str(row.get("label", "")).strip()
                if label not in classes:
                    continue
                crop_text = str(row.get("crop", "")).strip()
                if not crop_text:
                    continue
                path = Path(crop_text)
                if not path.is_absolute() and not path.exists():
                    candidate = dataset / path
                    if candidate.exists():
                        path = candidate
                if not path.exists():
                    candidate = REPO_ROOT / crop_text
                    if candidate.exists():
                        path = candidate
                if not path.exists():
                    continue
                source = str(row.get("source", "")).strip() or path.stem.split("__", 1)[0]
                result.append((path, label, source))

    if result:
        return result

    for label in classes:
        class_dir = dataset / "screw" / label
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                result.append((path, label, path.stem.split("__", 1)[0]))
    return result


def _predict(model, tensor: torch.Tensor, classes: list[str]):
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    index = int(np.argmax(probabilities))
    return classes[index], float(probabilities[index]), probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose screw classifier confidence and PIL/OpenCV preprocessing parity.")
    parser.add_argument("--dataset", default="artifacts/roi_dataset")
    parser.add_argument("--checkpoint", default="artifacts/screw_classifier/best.pt")
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all samples")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    model, checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()

    classes = list(checkpoint.get("class_names", DEFAULT_CLASSES))
    input_size = int(checkpoint.get("input_size", 224))
    mean = list(checkpoint.get("mean", IMAGENET_MEAN))
    std = list(checkpoint.get("std", IMAGENET_STD))
    pil_transform = build_transforms(input_size, train=False)

    samples = _collect_samples(dataset, classes)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No screw samples found in: {dataset}")

    source_count = len({source for _, _, source in samples})
    class_counts = {name: sum(1 for _, label, _ in samples if label == name) for name in classes}

    print("=== Screw Classifier Diagnostics ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(samples)} {class_counts}")
    print(f"Unique source images: {source_count}")
    if source_count < 5:
        print("WARNING: very few unique source images; generalization confidence cannot be trusted yet.")
    print("")
    print(f"{'File':<32} {'GT':<8} {'PIL pred/conf':<22} {'CV pred/conf':<22} {'|dP|max':>9}")
    print("-" * 100)

    pil_correct = 0
    cv_correct = 0
    pil_conf_correct: list[float] = []
    cv_conf_correct: list[float] = []
    tensor_diffs: list[float] = []
    probability_diffs: list[float] = []

    for path, label, _source in samples:
        with Image.open(path) as pil_image:
            pil_rgb = pil_image.convert("RGB")
            pil_tensor = pil_transform(pil_rgb).unsqueeze(0)

        image_bgr = read_image(path)
        cv_np = preprocess_opencv(image_bgr, input_size, mean, std)
        cv_tensor = torch.from_numpy(cv_np)

        tensor_diff = float(torch.max(torch.abs(pil_tensor - cv_tensor)).item())
        tensor_diffs.append(tensor_diff)

        pil_pred, pil_conf, pil_probs = _predict(model, pil_tensor, classes)
        cv_pred, cv_conf, cv_probs = _predict(model, cv_tensor, classes)
        prob_diff = float(np.max(np.abs(pil_probs - cv_probs)))
        probability_diffs.append(prob_diff)

        if pil_pred == label:
            pil_correct += 1
            pil_conf_correct.append(pil_conf)
        if cv_pred == label:
            cv_correct += 1
            cv_conf_correct.append(cv_conf)

        print(
            f"{path.name[:31]:<32} {label:<8} "
            f"{pil_pred + ' ' + format(pil_conf, '.4f'):<22} "
            f"{cv_pred + ' ' + format(cv_conf, '.4f'):<22} "
            f"{prob_diff:>9.6f}"
        )

    n = len(samples)
    pil_acc = pil_correct / n
    cv_acc = cv_correct / n
    max_tensor_diff = max(tensor_diffs) if tensor_diffs else 0.0
    max_prob_diff = max(probability_diffs) if probability_diffs else 0.0
    mean_pil_conf = float(np.mean(pil_conf_correct)) if pil_conf_correct else 0.0
    mean_cv_conf = float(np.mean(cv_conf_correct)) if cv_conf_correct else 0.0
    min_cv_conf = float(np.min(cv_conf_correct)) if cv_conf_correct else 0.0

    print("")
    print("=== Summary ===")
    print(f"PIL accuracy:            {pil_acc:.4f}")
    print(f"OpenCV accuracy:         {cv_acc:.4f}")
    print(f"Mean correct PIL conf:   {mean_pil_conf:.4f}")
    print(f"Mean correct OpenCV conf:{mean_cv_conf:.4f}")
    print(f"Min correct OpenCV conf: {min_cv_conf:.4f}")
    print(f"Max input tensor diff:   {max_tensor_diff:.6f}")
    print(f"Max probability diff:    {max_prob_diff:.6f}")

    print("")
    if max_prob_diff > 0.03:
        print("DIAGNOSIS: PIL/OpenCV preprocessing changes model output noticeably. Fix preprocessing parity before tuning confidence threshold.")
    elif cv_acc < 0.95:
        print("DIAGNOSIS: model does not fit even the available labeled ROI data reliably. Improve/retrain the classifier.")
    elif mean_cv_conf < 0.80:
        print("DIAGNOSIS: labels are mostly correct but the classifier is weakly separated/calibrated. More real source images and real missing-screw samples are needed before setting a production threshold.")
    else:
        print("DIAGNOSIS: classifier separation on the available ROI data looks reasonable. Use an independent calibration/validation set to choose the production confidence threshold.")


if __name__ == "__main__":
    main()
