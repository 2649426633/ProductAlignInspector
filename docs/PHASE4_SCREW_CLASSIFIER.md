# Phase 4 — screw / empty 小样本分类

这一阶段只解决一个问题：**对齐后的固定螺丝 ROI 中，当前状态到底是 `screw` 还是 `empty`。**

不让模型学习整张产品的位置，也不让模型定位螺丝。产品定位与 ROI 坐标已经由 Phase 1–3 解决。

## 模型

第一版使用 `MobileNetV3-Small` 二分类器：

```text
ROI
 ↓
PadToSquare
 ↓
224 × 224
 ↓
ImageNet normalization
 ↓
MobileNetV3-Small
 ↓
empty / screw
```

训练策略：

- ImageNet 预训练权重；
- 前 5 epoch 冻结 backbone，只训练分类头；
- 后续使用较低学习率微调整个网络；
- 对小样本使用轻量的旋转、平移、缩放、亮度/对比度、轻微模糊增强；
- 使用 class-weighted cross entropy 处理 `screw / empty` 数量不平衡；
- 优先按原始 source 图片分组切分 train/val，避免同一原图的不同 ROI 同时进入训练和验证；
- 最优模型按 validation macro-F1 保存。

## 1. 安装训练依赖

```bat
cd /d D:\Brunei
git pull --ff-only origin main
pip install -r requirements-train.txt
```

## 2. 确认数据目录

默认读取：

```text
artifacts/roi_dataset/
├── screw/
│   ├── screw/
│   │   ├── image001__S01.png
│   │   └── ...
│   └── empty/
│       ├── image001__E01.png
│       └── ...
└── manifest.csv
```

注意：从 GOOD 图片中的 `expected=empty` 孔位可以建立第一版 `empty` 类，但最终用于“漏装螺丝”的正式模型，最好补充几张**本来应该装螺丝但人为拆掉螺丝**的真实图片。否则模型可能学到“不同孔位的结构差别”，而不是真正的“有螺丝 / 无螺丝”差别。

## 3. 开始训练

默认参数：

```bat
python tools\train_screw_classifier.py --dataset "artifacts\roi_dataset" --output "artifacts\screw_classifier"
```

CPU 也可以：

```bat
python tools\train_screw_classifier.py --dataset "artifacts\roi_dataset" --output "artifacts\screw_classifier" --device cpu
```

有 CUDA 时 `--device auto` 会自动使用 CUDA。

第一版建议参数：

```bat
python tools\train_screw_classifier.py ^
  --dataset "artifacts\roi_dataset" ^
  --output "artifacts\screw_classifier" ^
  --epochs 30 ^
  --freeze-epochs 5 ^
  --batch-size 16 ^
  --input-size 224 ^
  --val-ratio 0.2
```

输出：

```text
artifacts/screw_classifier/
├── best.pt
├── training_history.csv
└── training_summary.json
```

终端重点看：

```text
All samples
Train / Val
Split
val_acc
val_f1
Best val macro-F1
```

如果出现：

```text
Split: sample_fallback_single_source
```

说明原始 source 图片太少，验证集无法做到 source 隔离。这种结果只能作为 smoke test，不能当工业准确率。

## 4. 导出 ONNX

```bat
python tools\export_screw_onnx.py ^
  --checkpoint "artifacts\screw_classifier\best.pt" ^
  --output "artifacts\screw_classifier\screw_classifier.onnx"
```

输出：

```text
screw_classifier.onnx
screw_classifier.json
```

导出脚本会同时：

1. ONNX checker 验证模型；
2. 用 ONNX Runtime 跑同一个随机输入；
3. 比较 PyTorch 与 ONNX 输出；
4. parity 误差大于 `1e-4` 时直接报错，不接受一个“能跑但结果已经变了”的 ONNX。

ONNX 输出不是 logits，而是：

```text
[empty_probability, screw_probability]
```

模型 metadata JSON 保存了 WinForms 后续必须严格复现的预处理：

```text
BGR → RGB
PadToSquare（白色）
Resize 224×224
/255.0
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
HWC → NCHW
```

## 5. 使用 ONNX Runtime 测一张真实 ROI

例如：

```bat
python tools\predict_screw_onnx.py ^
  --model "artifacts\screw_classifier\screw_classifier.onnx" ^
  --input "artifacts\roi_dataset\screw\screw\good__S01.png"
```

输出示例：

```text
Prediction: screw
Confidence: 0.991234
  empty: 0.008766
  screw: 0.991234
```

## 6. 小样本情况下怎么判断第一版是否值得继续

不要只看训练集 accuracy。

优先检查：

1. validation 是不是 `grouped_by_source`；
2. `empty` 和 `screw` 两类验证集都有真实图片；
3. 拿完全没有参与训练的原始图片重新对齐、裁 ROI，再测 ONNX；
4. 特别测试“应该有螺丝的位置，把螺丝真的拆掉”的图片。

如果模型对普通空孔很好，但对“漏装螺丝后的孔”判断不稳定，下一步不是换更大的模型，而是补少量真实 missing-screw ROI，再微调当前模型。

## 7. 交付目标

Phase 4/5 完成后，Python 侧最终需要交付给 WinForms 的螺丝模型只有：

```text
screw_classifier.onnx
screw_classifier.json
```

WinForms 不需要 Python、PyTorch、torchvision，也不需要 `best.pt`。
