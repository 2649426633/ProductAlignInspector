# ProductAlignInspector

工业装配完整性检测项目。当前第一阶段先解决一个基础问题：**产品在高分辨率图片中的位置不固定，但产品本体始终完整出现在画面内。**

目标流程：

```text
高分辨率原图
    ↓
产品自动定位
    ↓
粗对齐（平移 / 旋转 / 尺度）
    ↓
ECC 精配准到标准 GOOD 参考图
    ↓
标准坐标系
    ↓
固定 ROI
    ↓
螺丝 / 空孔 / 弹簧检测
    ↓
PASS / NG
```

## 当前阶段

Phase 1：产品定位与对齐。

这一阶段暂时不训练 YOLO、PatchCore 或分类网络。先把所有图片统一到同一个产品坐标系，后续螺丝和弹簧就可以在固定 ROI 内检测，从而显著降低对缺陷数据量的要求。

## 为什么这样做

当前图像特点：

- 产品一定完整出现在画面中；
- 产品 X/Y 位置不固定；
- 允许存在少量旋转；
- 背景大面积为亮色；
- 画面边缘可能存在暗角；
- 原图分辨率较高，小零件不应该先把整图强制缩到 640×640。

因此定位阶段采用 OpenCV 几何方法，而不是依赖神经网络学习位置变化。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. 创建标准参考图

选择一张结构正确、姿态较好的 GOOD 图片：

```bash
python tools/create_reference.py --input D:\data\good\good_001.png --output artifacts\reference
```

会输出：

```text
artifacts/reference/
├── reference_aligned.png
├── reference_mask.png
└── reference_meta.json
```

## 2. 对齐单张图片

```bash
python tools/align_image.py ^
  --input D:\data\test\test_001.png ^
  --reference artifacts\reference\reference_aligned.png ^
  --output artifacts\aligned
```

输出包含：

```text
aligned.png       # 最终标准化图像
coarse.png        # 粗对齐结果
foreground_mask.png
alignment.json    # 定位和 ECC 信息
overlay.png       # 与参考图叠加，方便肉眼检查
```

## 3. ROI 配置

`configs/product.example.json` 用来描述标准坐标中的螺丝孔位和弹簧区域。后续 WinForms 只需要加载产品配置，并在对齐图上裁固定 ROI。

## 后续计划

- Phase 1：产品自动定位 + ECC 配准 ✅（当前）
- Phase 2：ROI 标注工具（螺丝槽位、空孔、弹簧区域）
- Phase 3：螺丝 ROI 分类：`screw / empty`
- Phase 4：弹簧数量 / 缺失检测
- Phase 5：模型导出 ONNX / 产品特征导出 BIN
- Phase 6：C# .NET 8 + OpenCvSharp + ONNX Runtime 接入 WinForms

部署原则：**Python 用于开发、训练和模型导出；工业电脑端不依赖 Python，也不把 Python 打包成 exe。**
