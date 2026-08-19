# ProductAlignInspector

工业装配视觉检测项目。

## 当前固定主线

现在统一使用本地模板：

```text
artifacts/reference/brunei_preview_reference.png
```

它就是检测的 canonical 坐标系。当前流程固定为：

```text
RAW 原图（例如 5472x3648）
  ↓
SIFT similarity：统一坐标缩放 + 旋转 + X/Y 平移
  ↓
Euclidean ECC：只微调旋转 + X/Y 平移
  ↓
canonical 图（brunei_preview_reference.png 的尺寸）
  ↓
固定 ROI：S01/S02/E01...E09
  ↓
S / E / 弹簧 / 表面异常检测
  ↓
canonical_to_input 逆矩阵
  ↓
检测框映射回原始 RAW 图
  ↓
PASS / NG / ERROR + 日志
```

禁止使用非等比例 X/Y 拉伸、shear 或透视变换。

矫正失败不算产品 NG。失败帧必须：

```text
SKIP_ALIGNMENT
检测不运行
detection_run = false
写入日志
继续下一张
```

## 本地固定资源

```text
artifacts/reference/brunei_preview_reference.png
configs/brunei_preview.png
configs/brunei_preview_template.json
configs/brunei_preview_template_verify.png
artifacts/roi_dino_full/
dataset_roi_dino/
dataset_extra_ng/
```

这些大文件/数据保持本地，不提交到源码仓库。

## 主体源码

```text
product_align_inspector/
├─ alignment.py            RAW -> canonical 对齐，保存正/逆矩阵
├─ canonical_frame.py      canonical ROI/点 <-> RAW 坐标转换
├─ inspection_pipeline.py  主检测流水线，失败跳过并写日志
├─ decision_rules.py
├─ deploy_preprocess.py
├─ io_utils.py
├─ roi.py
├─ __init__.py
└─ anomaly/
   ├─ coreset.py
   ├─ dinov2_adapter.py
   ├─ roi_patchcore.py
   └─ __init__.py
```

## 先测试主体坐标链

当前还不接 S/E 检测器，先验证 101 张正式测试图片：

```bat
cd /d D:\Brunei

python tools\run_inspection_pipeline.py ^
  --input-root "D:\Brunei\dataset_roi_dino\test" ^
  --reference "D:\Brunei\artifacts\reference\brunei_preview_reference.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --output "D:\Brunei\artifacts\framework_final_test"
```

输出：

```text
artifacts/framework_final_test/
├─ canonical/             对齐后的 canonical 图片
├─ overlays/              canonical 与 reference 的 50/50 叠加 + ROI
├─ restored/              ROI 逆映射回原始 RAW 图
├─ inspection_log.jsonl
├─ inspection_summary.csv
└─ summary.json
```

`summary.json` 会统计本轮成功帧测得的 RAW->canonical uniform scale。等该 scale 在正常样本上稳定以后，再用：

```text
--canonical-scale <固定值>
```

把 coordinate scale 锁死。之后每一帧只允许旋转和 X/Y 平移变化。

## 后续模块顺序

```text
1. 主体对齐 + 坐标回映
2. S01/S02：应该有螺丝
3. E01~E09：应该为空
4. PASS/NG 规则
5. PatchCore 表面异常
6. WinForms + Hikrobot 相机
```

DINOv2 / PatchCore 保留用于划痕、磕伤、缺口、污渍和未知表面异常，不作为螺丝存在/缺失的主要判断器。

## 部署目标

开发/训练：Python。

最终 Windows 工业端：

```text
C# / .NET 8
OpenCvSharp
ONNX Runtime
Hikrobot MVS .NET SDK
```

最终工业电脑不依赖 Python 环境。
