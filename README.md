# ProductAlignInspector

工业装配视觉检测项目。

## 当前主线

```text
RAW 原图
  ↓
SIFT/RANSAC 仅用于寻找可靠对应点
  ↓
最终刚体矫正：rotation + X/Y translation，scale=1
  ↓
Euclidean ECC 微调：仍然只允许 rotation + X/Y translation
  ↓
canonical 图
  ↓
固定 ROI：S01/S02/E01...E09
  ↓
共享小型 CNN：输出 P(screw)
  ↓
11 个位置分别使用自己的 probability threshold
  ↓
S: P(screw) 低于阈值 => missing_screw
E: P(screw) 高于阈值 => excess_screw
  ↓
canonical_to_input 逆矩阵
  ↓
结果映射回 RAW
```

矫正失败不是产品 NG：记录 `SKIP_ALIGNMENT`，不执行检测，继续下一张。

## 为什么不再使用 GOOD-only 距离 bank

前面的 GOOD-bank / cosine-distance 路线只能回答“当前 ROI 像不像训练 GOOD”，不能稳定回答
“这里有没有螺丝”。跨拍摄批次、金属反光和局部亮度变化会让正常空孔距离异常增大，甚至出现
正常空孔分数高于真正多螺丝的情况。

因此 S/E 主线已切换到真正的监督二分类：

```text
empty / screw
```

模型只有一个，所有位置共享；阈值仍然是每个位置独立。

## 模型输出与 11 个阈值

共享 ONNX 只输出一个 logit：

```text
P(screw) = sigmoid(logit)
```

决策：

```text
S01 / S02:
  PASS  -> P(screw) >= 当前 S 位置 threshold
  NG    -> missing_screw

E01 ... E09:
  PASS  -> P(screw) <= 当前 E 位置 threshold
  NG    -> excess_screw
```

每个位置在 `model.json` 中保存：

- `threshold`
- `strict`
- `recommended`
- `loose`
- validation screw/empty probability 分布
- normal PASS rate / defect recall / balanced accuracy

`recommended` 默认选择验证集 balanced accuracy 最好的阈值。`strict` 优先缺陷召回，
`loose` 优先正常品通过率。

## 数据标注格式

训练使用宽表 CSV：

```text
image,source,split,S01,S02,E01,E02,...,E09
D:\...\good_01.bmp,good,,screw,screw,empty,empty,...,empty
```

合法标签：

```text
screw
empty
?
```

`?` 表示未知，训练时忽略，不允许程序猜具体缺陷位置。

### 自动生成初始 manifest

```bat
python tools\create_se_semantic_manifest.py ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --good-root "D:\...\train\good" ^
  --all-empty-root "D:\...\train\all_empty" ^
  --missing-root "D:\...\train\missing_screws" ^
  --excess-root "D:\...\train\excess_screws" ^
  --output "D:\Brunei\artifacts\se_semantic_manifest.csv"
```

生成规则：

```text
good:
  S=screw
  E=empty

all_empty:
  S=empty
  E=empty

missing_screws:
  E=empty
  S=?          # 不猜 S01/S02 哪个缺

excess_screws:
  S=screw
  E=?          # 不猜 E01..E09 哪个多
```

训练前把需要用于最终阈值标定的 `?` 改成真实的 `screw` 或 `empty`。

> 不建议把已经用于最终报告的 test 数据直接加入 train。若决定重新利用旧 test 数据，
> 应重新划分 train / val / final holdout，保留一批从未参与训练和阈值标定的最终测试图。

## 训练共享 CNN + 导出 ONNX

训练依赖：

```bat
pip install -r requirements-train.txt
```

训练：

```bat
python tools\build_se_presence_models.py ^
  --manifest "D:\Brunei\artifacts\se_semantic_manifest.csv" ^
  --reference "D:\Brunei\artifacts\reference\brunei_preview_reference.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --output "D:\Brunei\artifacts\se_classifier" ^
  --threshold-profile recommended
```

默认行为：

- image-level train/val split，避免同一原图的 ROI 同时泄漏到 train 和 val
- 一个共享 `TinyPresenceCNN`
- 输入 `1×96×96` 灰度
- CLAHE + per-crop 标准化
- 训练增强只有光照变化 + 小范围 X/Y jitter
- 不做 scale augmentation
- 不做 rotation augmentation
- `(slot,label)` 逆频率采样，避免位置/类别失衡
- early stopping
- ONNX 导出后自动检查 PyTorch/ONNX 数值一致性
- 11 个位置独立 threshold calibration

最终输出：

```text
artifacts/se_classifier/
├─ presence_classifier.onnx
├─ model.json
├─ metrics.json
├─ resolved_manifest.csv
└─ training_checkpoint.pt
```

最终 Windows 工业端只需要：

```text
presence_classifier.onnx
model.json
```

`training_checkpoint.pt` 只用于 Python 训练，不用于 C# 生产端。

## 训练数据覆盖要求

正式阈值标定要求每个位置的 validation 都至少出现：

```text
empty
screw
```

也就是说：

```text
S01/S02:
  需要 screw 正样本
  也需要 empty 缺失样本

E01...E09:
  需要 empty 正常样本
  也需要 screw 多余样本
```

如果某个位置没有两类样本，脚本默认停止，而不是给出虚假的“独立阈值”。

开发阶段可使用：

```text
--allow-incomplete-calibration
```

但这种全局 fallback 阈值只用于验证流水线是否能跑通，不作为最终生产阈值。

## 主流水线测试

Python runtime 需要：

```bat
pip install -r requirements.txt
```

运行：

```bat
python tools\run_inspection_pipeline.py ^
  --input-root "D:\Brunei\dataset_roi_dino\test\good" ^
  --reference "D:\Brunei\artifacts\reference\brunei_preview_reference.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --se-model "D:\Brunei\artifacts\se_classifier" ^
  --output "D:\Brunei\artifacts\se_semantic_test"
```

一个 `--se-model` 同时驱动：

```text
S_presence
E_empty
```

ONNX Session 只创建一次，两类 detector 共享。

日志中的 `score` 现在具有明确语义：

```text
score = probability_screw = P(screw)
```

不再是 GOOD-bank 距离。

## ROI 与几何约束

当前 Brunei canonical：

```text
3383 × 2071
```

ROI：

```text
S01, S02
E01 ... E09
```

每帧几何只允许：

```text
rotation
X translation
Y translation
```

禁止把 scale 当作每帧自由变量：

```text
scale = 1.0
```

CNN 的 `96×96` resize 只是固定分类器输入尺寸，不属于几何对齐。

## 主要源码

```text
product_align_inspector/
├─ alignment.py
├─ inspection_pipeline.py
├─ roi.py
├─ se_presence.py          # shared ONNX semantic runtime
├─ io_utils.py
└─ ...

tools/
├─ create_se_semantic_manifest.py
├─ build_se_presence_models.py
└─ run_inspection_pipeline.py
```

## 部署目标

开发 / 训练：

```text
Python
PyTorch
ONNX
```

Python 测试：

```text
OpenCV
ONNX Runtime
```

最终工业端：

```text
C# / .NET 8
OpenCvSharp
Microsoft.ML.OnnxRuntime
Hikrobot MVS .NET SDK
```

最终工业电脑不依赖 Python。

后续 PatchCore / DINO 只用于表面异常，不承担 S/E 螺丝存在性判断。
