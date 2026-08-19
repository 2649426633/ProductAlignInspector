# ProductAlignInspector

工业装配视觉检测项目。

## 当前主线

先稳定主体框架，不再保留已经放弃的 ResNet18 / 旧模板实验。

```text
RAW 图像
  ↓
几何矫正
  ├─ 失败：SKIP_ALIGNMENT，跳过检测并写日志
  ↓
统一检测坐标
  ↓
S / E / 弹簧 / 表面异常检测模块
  ↓
PASS / NG / ERROR
  ↓
结果映射回原图 + 检测日志
```

矫正失败不算产品 NG；它属于前处理失败，必须跳过后续检测并单独记录。

## 当前保留源码

```text
product_align_inspector/
├─ alignment.py
├─ canonical_frame.py
├─ decision_rules.py
├─ deploy_preprocess.py
├─ inspection_pipeline.py
├─ io_utils.py
├─ roi.py
├─ __init__.py
└─ anomaly/
   ├─ coreset.py
   ├─ dinov2_adapter.py
   ├─ roi_patchcore.py
   └─ __init__.py
```

## 当前保留工具

```text
tools/
├─ align_folder.py
├─ align_image.py
├─ annotate_rois.py
├─ build_roi_dino_patchcore.py
├─ evaluate_roi_dino_patchcore.py
├─ export_dinov2_onnx.py
├─ export_runtime_bundle.py
├─ extract_roi_dataset.py
├─ inspect_roi_dino_patchcore.py
├─ run_inspection_pipeline.py
├─ smoke_test_roi_dino.py
├─ verify_alignment_dataset.py
└─ verify_rois.py
```

旧的 RAW reference 创建工具、alignment recovery 临时诊断、ResNet18 螺丝分类器和 canonical round-trip 临时工具已经移除。

## 本地工作目录

大型数据、模型产物和参考图保持在本地，不提交到源码仓库。当前本地使用：

```text
artifacts/reference/brunei_preview_reference.png
artifacts/roi_dino_full/
configs/brunei_preview.png
configs/brunei_preview_template.json
configs/brunei_preview_template_verify.png
dataset_roi_dino/
dataset_extra_ng/
```

`artifacts/` 已由 `.gitignore` 忽略；数据集也保持本地。

## PatchCore

DINOv2 / PatchCore 保留用于后续表面异常，例如划痕、磕伤、缺口、污渍和未知异常，不再作为螺丝存在/缺失的主要判断器。

## 部署原则

开发/训练：Python。

最终 Windows 工业端目标：

```text
C# / .NET 8
OpenCvSharp
ONNX Runtime
Hikrobot MVS .NET SDK
```

最终工业电脑不依赖 Python 环境。
