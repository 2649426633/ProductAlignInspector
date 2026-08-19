# ProductAlignInspector

工业装配视觉检测项目。

## 当前主线

现在先把系统收敛到最简单的四步：

```text
原始高分辨率图片
    ↓
产品整体矫正到唯一标准参考图
    ↓
固定 ROI（S01/S02/E01...）
    ↓
ROI 检测
```

当前阶段只做前两步：**先把产品矫正和固定 ROI 坐标彻底稳定**。

在矫正通过之前，不继续叠加 ResNet18、模板匹配、局部搜索或其他分类实验。

## 唯一标准参考图

正式几何基准只使用：

```text
D:\Brunei\artifacts\reference\reference_aligned.png
```

ROI 配置必须是在这个参考图的同一坐标系中标注的。

`brunei_preview.png` 只用于人工确认 ROI 位置，不再把它处理成新的 reference，也不再做 flip/remap。

## 当前只需要的工具

```text
tools\create_reference.py          创建标准参考图
tools\align_image.py               单张产品矫正
tools\align_folder.py              批量产品矫正
tools\annotate_rois.py             在标准参考图上标固定 ROI
tools\verify_rois.py               单张：矫正 + 固定 ROI 可视化
tools\verify_alignment_dataset.py  批量：只验证矫正和固定 ROI
```

`verify_rois.py` 和 `verify_alignment_dataset.py` 都会检查 ROI 配置尺寸是否与 reference 一致；不一致时直接停止，不自动缩放、不自动 remap。

默认情况下，`foreground_quadrant` 这类无可靠 feature matrix 的 fallback 不算正式成功，应视为 RETRY。

## 第一步：确认 ROI 配置

你已经人工确认过的 ROI JSON 才能作为正式配置。

在正式回写 `configs\brunei.json` 前，必须满足：

```text
reference_width  == reference_aligned.png 宽度
reference_height == reference_aligned.png 高度
```

如果不一致，不要缩放坐标，先重新统一 reference / ROI 坐标系。

## 第二步：只验证 GOOD 的矫正

例如：

```bat
cd /d D:\Brunei
python tools\verify_alignment_dataset.py ^
  --input-root "D:\Brunei\dataset_roi_dino\test" ^
  --reference "D:\Brunei\artifacts\reference\reference_aligned.png" ^
  --config "D:\Brunei\configs\brunei_preview_template.json" ^
  --scenario good ^
  --output "D:\Brunei\artifacts\alignment_check_good"
```

输出：

```text
artifacts\alignment_check_good\
├─ aligned\
├─ overlays\
├─ alignment_summary.csv
└─ summary.json
```

这一步**没有任何螺丝识别**。

只检查每张 `overlays`：同一套 S01/S02/E01... 固定框是否始终准确落在相同物理位置。

目标：

```text
GOOD 01：ROI 正确
GOOD 02：ROI 正确
...
GOOD 10：ROI 正确
```

如果某张走 fallback，或者固定 ROI 明显偏位，只修 alignment，不调检测模型。

## 第三步：再做螺丝存在检测

只有当矫正稳定以后，再做：

```text
S01/S02：应该有螺丝
E01~E09：应该为空
```

这里的目标只是简单的 `screw present / empty`，不再同时混入对齐问题。

## 第四步：表面瑕疵

DINOv2 / PatchCore 继续保留，后面用于：

- 划痕
- 磕伤
- 缺口
- 污渍/异物
- 其他未知表面异常

它不再承担当前的产品几何矫正问题。

## 部署原则

开发/训练：Python。

最终 Windows 工业端：

```text
C# / .NET 8
OpenCvSharp
ONNX Runtime
```

工业电脑不依赖 Python 环境。
