# ProductAlignInspector

工业装配视觉检测项目。

## 当前固定主线

本地 canonical 模板：

```text
artifacts/reference/brunei_preview_reference.png
```

生产流程：

```text
RAW 原图
  ↓
SIFT/RANSAC 找可靠对应点
  ↓
最终刚体矫正：rotation + X/Y translation，scale=1
  ↓
Euclidean ECC 微调：仍然只允许 rotation + X/Y translation
  ↓
canonical 图
  ↓
固定 ROI 坐标映射
  ↓
S_presence + E_empty 两个独立检测器
  ↓
canonical_to_input 逆矩阵
  ↓
检测框映射回 RAW 原图
  ↓
PASS / NG / ERROR + 日志
```

矫正失败不算产品 NG：`SKIP_ALIGNMENT`、不执行检测、写日志、继续下一张。

## ROI 坐标

`brunei_preview_template.json` 中的 ROI 可以保持原来的标注坐标系。若 ROI JSON 的
`coordinate_system.image_width/image_height` 与当前 canonical 图片尺寸不同，系统只做一次
固定坐标换算，不会把它当成每张图片的几何变化。

## S / E 两个独立模型

S 与 E 不共用模型：

```text
S_presence
  S01/S02 正常状态 = 有螺丝
  异常 = missing_screw

E_empty
  E01...E09 正常状态 = 空
  异常 = excess_screw
```

当前先使用轻量 GOOD-only 外观 bank：CLAHE + 灰度结构 + Scharr 梯度，最近邻余弦距离。
它不依赖 PyTorch，后面容易迁移到 OpenCvSharp/C#。PatchCore 继续保留给表面异常，不承担螺丝存在/缺失。

### 1. 用 50 张 train/good 建 S 和 E 两套模型

```bat
python tools\build_se_presence_models.py ^
  --train-good "D:\Brunei\dataset_roi_dino\train\good" ^
  --reference "D:\Brunei\artifacts\reference\brunei_preview_reference.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --output "D:\Brunei\artifacts\se_presence"
```

输出：

```text
artifacts/se_presence/
├─ S/
│  ├─ model.json
│  └─ banks/S01.npy, S02.npy
├─ E/
│  ├─ model.json
│  └─ banks/E01.npy ... E09.npy
└─ build_summary.json
```

### 2. 接入主流水线测试 101 张

```bat
python tools\run_inspection_pipeline.py ^
  --input-root "D:\Brunei\dataset_roi_dino\test" ^
  --reference "D:\Brunei\artifacts\reference\brunei_preview_reference.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --s-model "D:\Brunei\artifacts\se_presence\S" ^
  --e-model "D:\Brunei\artifacts\se_presence\E" ^
  --output "D:\Brunei\artifacts\se_detection_test"
```

`overlays/` 显示 canonical 上的 PASS/NG ROI；`restored/` 把同一检测结果映射回 RAW 原图。
`inspection_summary.csv` 会分别记录 `S_status` 和 `E_status`。

## 主体源码

```text
product_align_inspector/
├─ alignment.py
├─ canonical_frame.py
├─ inspection_pipeline.py
├─ se_presence.py
├─ roi.py
├─ decision_rules.py
├─ deploy_preprocess.py
├─ io_utils.py
├─ __init__.py
└─ anomaly/
```

## 后续模块顺序

```text
1. 主体对齐 + 坐标回映
2. S_presence / E_empty 基线验证
3. 根据 101 张结果分别调 S 与 E 阈值/特征
4. PASS/NG 规则
5. PatchCore 表面异常
6. WinForms + Hikrobot 相机
```

## 部署目标

开发/训练：Python。

最终 Windows 工业端：

```text
C# / .NET 8
OpenCvSharp
ONNX Runtime（需要时）
Hikrobot MVS .NET SDK
```

最终工业电脑不依赖 Python 环境。
