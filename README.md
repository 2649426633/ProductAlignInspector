# ProductAlignInspector

工业装配完整性检测项目。当前目标是：产品在高分辨率图片中的位置可以变化，但产品始终完整出现在画面内；系统先自动对齐到标准坐标，再检查螺丝、空孔、弹簧等固定装配位置。

目标流程：

```text
高分辨率原图
    ↓
产品自动定位 / SIFT 配准
    ↓
ECC 小范围精配准
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

- Phase 1：产品自动定位与对齐 ✅
- Phase 2：ROI 标注 ✅ 当前进行

先把所有图片统一到同一个产品坐标系，再在标准参考图上定义固定 ROI。这样后续螺丝和弹簧检测不需要学习产品在整张图里的位置变化，可以显著减少缺陷数据需求。

## 安装

Windows PowerShell / CMD：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

也可以只安装依赖：

```bash
pip install -r requirements.txt
```

`tools/` 下脚本已经支持直接运行，即使没有先执行 `pip install -e .` 也能找到项目包。

## 1. 创建标准参考图

例如：

```bat
python tools\create_reference.py --input "D:\Brunei\good.bmp" --output "artifacts\reference"
```

输出：

```text
artifacts/reference/
├── reference_aligned.png
├── reference_mask.png
└── reference_meta.json
```

## 2. 对齐单张图片

```bat
python tools\align_image.py --input "D:\Brunei\test.bmp" --reference "artifacts\reference\reference_aligned.png" --output "artifacts\test_align"
```

输出：

```text
aligned.png
coarse.png
foreground_mask.png
alignment.json
overlay.png
```

`overlay.png` 用来肉眼检查标准参考图与当前图片是否真正重合。

## 3. 批量对齐数据集

```bat
python tools\align_folder.py --input-dir "D:\Brunei\dataset" --reference "artifacts\reference\reference_aligned.png" --output-dir "artifacts\aligned_dataset"
```

脚本会递归处理子目录，并生成 `alignment_report.csv`。单张失败不会中断整个批处理。

## 4. ROI 标注

打开标准参考图：

```bat
python tools\annotate_rois.py --image "artifacts\reference\reference_aligned.png" --output "configs\brunei.json" --product "Brunei"
```

操作方式：

```text
鼠标左键拖动  画一个 ROI
S             保存为“这里应该有螺丝”
E             保存为“这里应该为空”
P             保存为弹簧区域，并在终端输入期望弹簧数量
C             取消当前未保存 ROI
U             撤销
W             保存 JSON + 预览图
Q / ESC       退出
```

例如一个螺丝位：先用鼠标框住螺丝及周围少量背景，然后按 `S`。

如果某个孔标准状态应该没有螺丝：框住该孔后按 `E`。

弹簧建议先框“整个需要计数的弹簧区域”，然后按 `P`，终端会要求输入标准弹簧数量。

保存后得到：

```text
configs/brunei.json
configs/brunei_preview.png
```

所有 ROI 坐标都使用 `reference_aligned.png` 的原始像素坐标，不使用缩小后的屏幕坐标。

## ROI 配置示例

```json
{
  "schema_version": 1,
  "product": "Brunei",
  "coordinate_system": {
    "reference_image": ".../reference_aligned.png",
    "image_width": 3200,
    "image_height": 1800
  },
  "screw_slots": [
    {
      "id": "S01",
      "roi": [1000, 500, 180, 180],
      "expected": "screw",
      "enabled": true
    },
    {
      "id": "E01",
      "roi": [1400, 500, 180, 180],
      "expected": "empty",
      "enabled": true
    }
  ],
  "spring_regions": [
    {
      "id": "SPRING01",
      "roi": [300, 600, 900, 500],
      "expected_count": 8,
      "enabled": true
    }
  ]
}
```

## 当前代码结构

```text
ProductAlignInspector/
├── product_align_inspector/
│   ├── alignment.py       # SIFT/前景定位、粗配准、ECC 精配准
│   ├── io_utils.py        # Windows 中文路径安全读写
│   └── roi.py             # ROI 配置加载与裁切
├── tools/
│   ├── create_reference.py
│   ├── align_image.py
│   ├── align_folder.py
│   └── annotate_rois.py   # 鼠标 ROI 标注
├── configs/
│   └── product.example.json
├── pyproject.toml
└── requirements.txt
```

## 后续计划

- Phase 1：产品自动定位 + 配准 ✅
- Phase 2：ROI 标注工具 ✅
- Phase 3：从正常/模拟缺陷图片自动生成 `screw / empty` ROI 数据集
- Phase 4：训练小型螺丝 ROI 分类模型并导出 ONNX
- Phase 5：弹簧数量 / 缺失检测
- Phase 6：完整 Python inspection pipeline + PASS/NG
- Phase 7：ONNX / BIN 交付格式
- Phase 8：C# .NET 8 + OpenCvSharp + ONNX Runtime 接入 WinForms

部署原则：**Python 用于开发、训练、验证和模型导出；工业电脑端不依赖 Python，也不把 Python 打包成 exe。**
