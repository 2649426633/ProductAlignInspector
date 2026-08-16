# Phase 5 — 完整螺丝检测 Pipeline

这一阶段把前面的模块串起来：

```text
原始生产图片
  ↓
SIFT / ECC 产品配准
  ↓
标准坐标系
  ↓
按 brunei.json 裁固定螺丝 ROI
  ↓
MobileNetV3-Small screw / empty 分类
  ↓
与 expected 状态比较
  ↓
Missing Screw / Extra Screw / Low Confidence
  ↓
PASS / NG
```

当前 Python 验证默认直接加载 `best.pt`，原因是部分 Windows/Anaconda 环境存在 Python `onnxruntime` DLL 初始化问题。最终 WinForms 部署仍使用 `screw_classifier.onnx`。

## 运行

```bat
cd /d D:\Brunei
git pull --ff-only origin main

python tools\inspect_product.py ^
  --input "D:\Brunei\test.bmp" ^
  --reference "artifacts\reference\reference_aligned.png" ^
  --config "configs\brunei.json" ^
  --checkpoint "artifacts\screw_classifier\best.pt" ^
  --output "artifacts\inspection_test"
```

默认置信度阈值是 `0.80`。低于阈值时即使预测类别与 expected 相同，也会按 `Low Confidence` 判为 NG，以避免工业现场把不确定结果当 PASS。

可以调整：

```bat
--confidence 0.90
```

## 输出

```text
artifacts/inspection_test/
├── aligned.png
├── inspection_preview.png
├── inspection.json
└── crops/
    ├── S01.png
    ├── S02.png
    └── ...
```

终端会显示每个位置：

```text
ID         Expected   Actual         Conf Result
--------------------------------------------------------------
S01        screw      screw        0.9921 PASS
S02        screw      empty        0.9814 NG:Missing Screw
E01        empty      empty        0.9977 PASS

FINAL: NG
```

`inspection_preview.png` 会在标准对齐图上直接画框：

- 绿色：该位置通过；
- 红色：状态错误；
- 黄色：模型置信度低；
- 顶部显示最终 `PASS / NG`。

## 判定规则

```text
expected=screw + actual=screw  → PASS
expected=screw + actual=empty  → Missing Screw → NG
expected=empty + actual=empty  → PASS
expected=empty + actual=screw  → Extra Screw → NG
confidence < threshold         → Low Confidence → NG
alignment quality too low      → NG
```

## Python / ONNX / C# 预处理契约

Phase 5 不再使用 PIL 作为推理预处理。`inspect_product.py` 使用与 ONNX/C# 约定一致的 OpenCV 路径：

```text
BGR ROI
→ PadToSquare，白色 255
→ Resize 224×224，INTER_LINEAR
→ BGR → RGB
→ /255.0
→ ImageNet mean/std
→ HWC → CHW
→ float32 NCHW
```

后续 C# + OpenCvSharp 必须逐项复制同样步骤，从而保证 Python / ONNX / WinForms 输入一致。

## 当前范围

这一阶段只判定螺丝/空孔。弹簧区域已经保留在产品配置和最终 JSON 中，但标记为 `not_evaluated_in_this_phase`，下一阶段再实现弹簧数量/缺失检测。
