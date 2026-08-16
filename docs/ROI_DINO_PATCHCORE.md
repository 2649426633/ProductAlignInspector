# Alignment + Fixed ROI + DINOv2 / PatchCore

本路线用于 **GOOD 多、NG 少** 的工业装配/表面异常检测。

旧仓库 `2649426633/patchcores` 中的 DINOv2 patch-token 提取和 PatchCore coreset 思路已经迁移并适配到本项目。新架构不再让 PatchCore 在整张高分辨率图中寻找产品或异常 ROI，而是复用 ProductAlignInspector 已经验证成功的产品配准和固定 ROI。

```text
5472x3648 原图
      ↓
SIFT + ECC 产品配准
      ↓
标准坐标系
      ↓
固定 ROI: S01 / S02 / SURFACE01 / ...
      ↓
多个 ROI 一次 batch 输入 DINOv2-S/14
      ↓
每个 ROI 的 16x16 patch tokens (224 输入)
      ↓
每个 ROI 独立 GOOD Memory Bank + coreset
      ↓
最近邻 cosine distance
      ↓
局部 anomaly map + ROI anomaly score
      ↓
阈值
      ↓
PASS / NG
```

## 为什么和旧项目不同

旧 `PatchCoreDINOv2Pipeline` 的结构主要是：

```text
整图 PatchCore -> 找异常 bbox -> 裁异常 ROI -> DINOv2 缺陷样本匹配
```

新项目已经能把产品稳定对齐，所以第一阶段改成：

```text
产品配准 -> 固定 ROI -> DINOv2 patch tokens -> PatchCore-style GOOD Memory Bank
```

这样有几个直接好处：

- 不需要让异常模型学习产品在整图中的位置变化；
- 不需要 NG 图片训练第一版；
- 小 ROI 计算量明显低于整张 20MP 图片 PatchCore；
- S01、S02、表面区可以各自拥有独立 Memory Bank 和阈值；
- 后续容易导出 DINOv2 ONNX，并把 Memory Bank 保存为 C# 可直接读取的 BIN。

## 1. DINOv2 文件

不把 DINOv2 官方仓库和大权重提交到 ProductAlignInspector GitHub。

如果旧项目本机仍然有：

```text
D:\wlenai\third_party\dinov2
D:\wlenai\weights\dinov2_vits14_pretrain.pth
```

新项目可以直接通过命令行引用，不需要复制。

也可以自行放到：

```text
D:\Brunei\third_party\dinov2
D:\Brunei\weights\dinov2_vits14_pretrain.pth
```

## 2. 安装依赖

```bat
cd /d D:\Brunei
git pull --ff-only origin main
pip install -r requirements-anomaly.txt
```

如果当前 PyTorch/CUDA 环境已经可以运行旧 DINOv2，不建议为了本项目重新安装另一套 CUDA PyTorch。

## 3. 数据集怎么准备

### 训练数据只放 GOOD 完整原图

例如：

```text
D:\Brunei\dataset_roi_dino\
├─ train\
│  └─ good\
│     ├─ good_001.bmp
│     ├─ good_002.bmp
│     ├─ good_003.bmp
│     └─ ...
│
└─ test\
   ├─ good\
   │  ├─ normal_001.bmp
   │  └─ ...
   └─ ng\
      ├─ missing_screw_001.bmp
      ├─ scratch_001.bmp
      └─ ...
```

`train/good` 中必须是正常产品的 **完整相机原图**，不要人工裁 ROI。程序自己完成：

```text
完整 GOOD 原图 -> 配准 -> 固定 ROI -> DINO 特征 -> Memory Bank
```

建议：

- smoke test：10 张不同 GOOD 原图；
- 第一版：30~50 张 GOOD；
- 后续生产：逐步补充不同批次、正常位置波动、正常光照波动。

不要复制同一张图片来凑数量。

`test/ng` 不参与 Memory Bank 建立，只用于验证漏螺丝、错装、划痕等异常是否能把 score 拉高。

## 4. 第一阶段先训练 S01 / S02

假设当前 `configs/brunei.json` 已经有 S01/S02：

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

如果 DINO 文件已经复制到 `D:\Brunei\third_party` / `weights`，则 `--dino-repo` 和 `--dino-weights` 可以省略。

### 输出

```text
artifacts/roi_dino_patchcore/
├─ model.json
├─ build_report.csv
└─ banks/
   ├─ S01.npz
   └─ S02.npz
```

每个 `.npz` 都是独立的正常 Memory Bank。

模型建立时会把 GOOD 原图按 source 分为：

```text
80% -> 建 Memory Bank
20% -> GOOD 阈值校准
```

默认阈值：

```text
max(calibration GOOD score) x 1.10
```

因此第一版阈值不是手工拍脑袋设置。

## 5. 测试一张原始图片

```bat
python tools\inspect_roi_dino_patchcore.py ^
  --input "D:\Brunei\test.bmp" ^
  --model-dir "artifacts\roi_dino_patchcore" ^
  --dino-repo "D:\wlenai\third_party\dinov2" ^
  --dino-weights "D:\wlenai\weights\dinov2_vits14_pretrain.pth" ^
  --device auto ^
  --output "artifacts\roi_dino_test"
```

输出：

```text
artifacts/roi_dino_test/
├─ aligned.png
├─ inspection_preview.png
├─ inspection.json
├─ crops/
│  ├─ S01.png
│  └─ S02.png
└─ heatmaps/
   ├─ S01.png
   └─ S02.png
```

终端会显示：

```text
ROI                 Score  Threshold   MaxPatch Result
--------------------------------------------------------------
S01              0.012345   0.024500   0.041200 PASS
S02              0.087100   0.026300   0.190200 NG
```

这里的 `Score` 不是分类概率，而是该 ROI 相对于正常 Memory Bank 的异常距离；越大越异常。

## 6. 表面划痕/磕伤 ROI

除了 `screw_slots` 和 `spring_regions`，配置现在还支持：

```json
"anomaly_regions": [
  {
    "id": "SURFACE01",
    "roi": [900, 300, 500, 350],
    "enabled": true
  }
]
```

之后可以对 `SURFACE01` 单独建立正常 Memory Bank。

对于非常细小的表面缺陷，224 输入可能不够。DINOv2-S/14 可以使用 448 输入：

```bat
--image-size 448
```

448 会产生 32x32 patch grid，局部空间分辨率更高，但 DINO 推理时间和 Memory Bank 特征数量也会明显增加。第一轮建议先用 224 验证路线，再针对细划痕 ROI 升到 448。

## 7. 当前版本的部署方向

当前 Python 版本用于验证算法。

验证通过后目标交付：

```text
dinov2_vits14.onnx
S01_memory.bin
S02_memory.bin
SURFACE01_memory.bin
product_model.json
```

WinForms：

```text
OpenCvSharp 配准/裁 ROI
       ↓
Microsoft.ML.OnnxRuntime 跑 DINOv2
       ↓
C# cosine/L2 最近邻 Memory Bank
       ↓
ROI score / threshold
       ↓
PASS / NG
```

工业电脑端不需要 Python。

## 8. 当前最重要的验证

第一轮不要同时做几十个 ROI。先只做 S01/S02：

1. 用 10~30 张 GOOD 建 bank；
2. 用完全不同的 GOOD 图片测试，score 应稳定低于 threshold；
3. 拆掉 S01 螺丝拍一张，S01 score 应明显升高；
4. 恢复 S01、拆掉 S02，S02 score 应明显升高；
5. 如果 GOOD 和 missing screw 的 score 有明显间隔，再扩展到其他 ROI 和表面瑕疵。

这一步成功后再进入 ONNX/BIN 和 C#，不会提前为了部署复杂化算法验证。
