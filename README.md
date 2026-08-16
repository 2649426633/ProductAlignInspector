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
- Phase 2：ROI 标注 ✅
- Phase 3：ROI 稳定性验证 + 训练数据自动裁切 ✅ 当前

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

保存后得到：

```text
configs/brunei.json
configs/brunei_preview.png
```

所有 ROI 坐标都使用 `reference_aligned.png` 的原始像素坐标，不使用缩小后的屏幕坐标。

## 5. 在 test.bmp 上验证 ROI 是否稳定

在进入训练前，先确认对齐后每个固定 ROI 仍然准确覆盖同一个螺丝孔/空孔/弹簧区域：

```bat
python tools\verify_rois.py --input "D:\Brunei\test.bmp" --reference "artifacts\reference\reference_aligned.png" --config "configs\brunei.json" --output "artifacts\verify_test"
```

重点打开：

```text
artifacts/verify_test/roi_preview.png
```

并检查：

```text
artifacts/verify_test/crops/screw_slots/
artifacts/verify_test/crops/spring_regions/
```

如果 `roi_preview.png` 中所有框都准确落在对应装配位置，说明“对齐 → 固定 ROI”链路成立，可以开始批量生成训练数据。

## 6. 从 GOOD 图片批量生成 ROI 数据集

推荐只把确认正常的 GOOD 图片作为这一阶段输入，不要把未知 test/NG 图片混入训练集。

单张 GOOD 测试：

```bat
python tools\extract_roi_dataset.py --input "D:\Brunei\good.bmp" --reference "artifacts\reference\reference_aligned.png" --config "configs\brunei.json" --output "artifacts\roi_dataset"
```

如果以后 GOOD 图片放在目录中：

```bat
python tools\extract_roi_dataset.py --input-dir "D:\Brunei\good" --reference "artifacts\reference\reference_aligned.png" --config "configs\brunei.json" --output "artifacts\roi_dataset"
```

输出结构：

```text
artifacts/roi_dataset/
├── screw/
│   ├── screw/
│   │   ├── image001__S01.png
│   │   └── ...
│   └── empty/
│       ├── image001__E01.png
│       └── ...
├── spring/
│   ├── count_8/
│   │   └── image001__SPRING01.png
│   └── ...
└── manifest.csv
```

`manifest.csv` 会保存源图、ROI ID、标签、配准方法、SIFT 匹配数、inlier ratio、ECC 分数和失败原因，后续训练前可以先过滤配准质量不好的图片。

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
│   ├── alignment.py
│   ├── io_utils.py
│   └── roi.py
├── tools/
│   ├── create_reference.py
│   ├── align_image.py
│   ├── align_folder.py
│   ├── annotate_rois.py
│   ├── verify_rois.py
│   └── extract_roi_dataset.py
├── configs/
│   └── product.example.json
├── pyproject.toml
└── requirements.txt
```

## 后续计划

- Phase 1：产品自动定位 + 配准 ✅
- Phase 2：ROI 标注工具 ✅
- Phase 3：ROI 稳定性验证 + GOOD ROI 数据集生成 ✅
- Phase 4：建立缺螺丝/多螺丝等模拟缺陷采集流程，并训练小型 `screw / empty` 模型
- Phase 5：导出螺丝分类 ONNX
- Phase 6：弹簧数量 / 缺失检测
- Phase 7：完整 Python inspection pipeline + PASS/NG
- Phase 8：ONNX / BIN 交付格式
- Phase 9：C# .NET 8 + OpenCvSharp + ONNX Runtime 接入 WinForms

部署原则：**Python 用于开发、训练、验证和模型导出；工业电脑端不依赖 Python，也不把 Python 打包成 exe。**
