from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image


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


def preprocess(image_bgr: np.ndarray, metadata: dict) -> np.ndarray:
    preprocess_cfg = metadata["preprocess"]
    input_size = int(metadata["input_size"])

    image = image_bgr
    if bool(preprocess_cfg.get("pad_to_square", True)):
        pad_values = preprocess_cfg.get("pad_value_rgb", [255, 255, 255])
        pad_value = int(pad_values[0])
        image = _pad_to_square(image, pad_value)

    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.asarray(preprocess_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(preprocess_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image[None, ...], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX screw/empty inference on one ROI crop.")
    parser.add_argument("--model", default="artifacts/screw_classifier/screw_classifier.onnx")
    parser.add_argument("--meta", help="Model metadata JSON; defaults to ONNX path with .json")
    parser.add_argument("--input", required=True, help="One screw-slot ROI crop")
    args = parser.parse_args()

    model_path = Path(args.model)
    metadata_path = Path(args.meta) if args.meta else model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    image = read_image(args.input)
    tensor = preprocess(image, metadata)

    input_name = str(metadata.get("input_name", session.get_inputs()[0].name))
    probabilities = session.run(None, {input_name: tensor})[0][0]
    classes = list(metadata["classes"])

    best_index = int(np.argmax(probabilities))
    print(f"Prediction: {classes[best_index]}")
    print(f"Confidence: {float(probabilities[best_index]):.6f}")
    for index, name in enumerate(classes):
        print(f"  {name}: {float(probabilities[index]):.6f}")


if __name__ == "__main__":
    main()
