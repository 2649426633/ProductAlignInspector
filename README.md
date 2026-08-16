# ProductAlignInspector

工业装配完整性与局部瑕疵检测项目。

当前主路线已经切换为：**产品矫正 + 固定 ROI + DINOv2 / PatchCore-style GOOD Memory Bank**。

```text
高分辨率原图
    ↓
SIFT + ECC 产品配准
    ↓
标准坐标系
    ↓
固定 ROI
    ↓
DINOv2 patch tokens
    ↓
每个 ROI 独立 GOOD Memory Bank + coreset
    ↓
局部 anomaly map / anomaly score
    ↓
PASS / NG
```

这条路线主要针对：

- GOOD 数据多、真实 NG/瑕疵样本少；
- 漏螺丝、多螺丝、错装、部件缺失；
- 螺丝局部异常；
- 划痕、磕伤、缺口、异物等未预先分类的局部异常。

之前的 `screw / empty` MobileNet 分类代码仍保留作为监督分类基线，但当前不再作为唯一主方案。

## 当前阶段

- Phase 1：产品自动定位与配准 ✅
- Phase 2：固定 ROI 标注与稳定性验证 ✅
- Phase 3：MobileNet `screw / empty` 基线 ✅
- Phase 4：ROI DINOv2 / PatchCore GOOD-only Memory Bank ✅ 已开始
- Phase 5：真实 GOOD / NG score separation 验证 ← 当前
- Phase 6：DINOv2 ONNX + ROI Memory Bank BIN
- Phase 7：C# .NET 8 + OpenCvSharp + ONNX Runtime

## 安装基础依赖

```bat
cd /d D:\Brunei
pip install -r requirements.txt
```

ROI DINO/PatchCore 开发依赖：

```bat
pip install -r requirements-anomaly.txt
```

## 1. 创建标准参考图

```bat
python tools\create_reference.py --input "D:\Brunei\good.bmp" --output "artifacts\reference"
```

## 2. 验证产品配准

```bat
python tools\align_image.py ^
  --input "D:\Brunei\test.bmp" ^
  --reference "artifacts\reference\reference_aligned.png" ^
  --output "artifacts\test_align"
```

重点检查：

```text
artifacts/test_align/overlay.png
```

## 3. 标注固定 ROI

```bat
python tools\annotate_rois.py ^
  --image "artifacts\reference\reference_aligned.png" ^
  --output "configs\brunei.json" ^
  --product "Brunei"
```

现有 `screw_slots`、`spring_regions` 都可以直接作为 anomaly ROI 使用。

另外配置支持通用表面区域：

```json
"anomaly_regions": [
  {
    "id": "SURFACE01",
    "roi": [900, 300, 500, 350],
    "enabled": true
  }
]
```

## 4. 准备 GOOD-only 数据

训练 Memory Bank 时只需要正常完整原图：

```text
D:\Brunei\dataset_roi_dino\
├─ train\
│  └─ good\
│     ├─ good_001.bmp
│     ├─ good_002.bmp
│     └─ ...
└─ test\
   ├─ good\
   └─ ng\
```

`train/good` 不需要人工裁 ROI。程序会自己完成配准和 ROI 裁剪。

建议第一轮先准备 10~30 张不同 GOOD 原图。

## 5. 建 S01 / S02 的 DINOv2 PatchCore Memory Bank

如果继续复用旧项目本地 DINOv2：

```text
D:\wlenai\third_party\dinov2
D:\wlenai\weights\dinov2_vits14_pretrain.pth
```

执行：

```bat
python tools\build_roi_dino_patchcore.py ^
  --good-dir "D:\Brunei\dataset_roi_dino\train\good" ^
  --reference "artifacts\reference\reference_aligned.png" ^
  --config "configs\brunei.json" ^
  --output "artifacts\roi_dino_patchcore" ^
  --roi-id S01 ^
  --roi-id S02 ^
  --dino-repo "D:\wlenai\third_party\dinov2" ^
  --dino-weights "D:\wlenai\weights\dinov2_vits14_pretrain.pth" ^
  --device auto ^
  --image-size 224 ^
  --coreset 0.10
```

输出：

```text
artifacts/roi_dino_patchcore/
├─ model.json
├─ build_report.csv
└─ banks/
   ├─ S01.npz
   └─ S02.npz
```

程序默认：

```text
80% GOOD source -> Memory Bank
20% GOOD source -> threshold calibration
```

## 6. 测试原始图片

```bat
python tools\inspect_roi_dino_patchcore.py ^
  --input "D:\Brunei\test.bmp" ^
  --model-dir "artifacts\roi_dino_patchcore" ^
  --dino-repo "D:\wlenai\third_party\dinov2" ^
  --dino-weights "D:\wlenai\weights\dinov2_vits14_pretrain.pth" ^
  --device auto ^
  --output "artifacts\roi_dino_test"
```

输出包括：

```text
aligned.png
inspection_preview.png
inspection.json
crops/
heatmaps/
```

`Score` 越高表示越偏离该 ROI 的正常 Memory Bank。

## ROI DINO/PatchCore 代码

```text
product_align_inspector/anomaly/
├─ dinov2_adapter.py   # 从旧 patchcores 项目迁移并改成 ROI batch patch-token 提取
├─ coreset.py          # PatchCore-style approximate greedy coreset
└─ roi_patchcore.py    # 每个固定 ROI 独立 Memory Bank / score / threshold

tools/
├─ build_roi_dino_patchcore.py
└─ inspect_roi_dino_patchcore.py
```

详细说明：`docs/ROI_DINO_PATCHCORE.md`。

## 第一轮验证目标

先只验证 S01/S02，不要一开始同时做所有区域：

```text
不同 GOOD 图片
S01 score 低
S02 score 低

拆掉 S01
S01 score 明显升高

恢复 S01、拆掉 S02
S02 score 明显升高
```

只要 GOOD 和异常 score 能形成明显间隔，再扩展到 E01/E02/E03、弹簧和表面瑕疵 ROI。

## 最终部署原则

Python 用于训练、验证和导出。

工业 Windows 端目标：

```text
OpenCvSharp              -> 产品配准 / ROI
Microsoft.ML.OnnxRuntime -> DINOv2 ONNX
*.bin                     -> ROI Memory Bank
C# cosine/L2              -> anomaly score
```

工业电脑不依赖 Python，也不把 Python 打包成 exe。
