# Screw / Empty 真实采集方案

当前只有一张原图时，即使能裁出多个 ROI，也不代表模型见过真实的拍摄变化。正式训练必须增加独立 source 图片，并且让“应该装螺丝的位置”真实出现 empty 状态。

推荐目录：

```text
D:\Brunei\captures\
├── normal\
│   ├── 001.bmp
│   ├── 002.bmp
│   └── ...
├── missing_S01\
│   ├── 001.bmp
│   ├── 002.bmp
│   └── ...
└── missing_S02\
    ├── 001.bmp
    ├── 002.bmp
    └── ...
```

含义：

- `normal`：产品正常装配；标签按 `configs/brunei.json` 的 expected 自动生成。
- `missing_S01`：人为拆掉 S01 螺丝；S01 自动标为 `empty`，其他位置仍按配置标签。
- `missing_S02`：人为拆掉 S02 螺丝；S02 自动标为 `empty`。
- 如果以后要检测“本应为空却多装螺丝”，可使用 `extra_E01`，E01 会自动标为 `screw`。
- 多个缺失位置可用 `missing_S01+missing_S02`。

第一轮不需要几千张。重点是增加真实 source 多样性。每种状态可先拍十几张作为工程基线：每拍一张，把产品重新放置一次，允许自然出现少量 X/Y 位移和小角度变化；不要通过复制同一张图片来凑数量。

提取数据：

```bat
python tools\extract_screw_scenarios.py ^
  --input-dir "D:\Brunei\captures" ^
  --reference "artifacts\reference\reference_aligned.png" ^
  --config "configs\brunei.json" ^
  --output "artifacts\screw_dataset_v2"
```

然后训练：

```bat
python tools\train_screw_classifier.py ^
  --dataset "artifacts\screw_dataset_v2" ^
  --output "artifacts\screw_classifier_v2" ^
  --epochs 30 ^
  --freeze-epochs 5 ^
  --batch-size 16 ^
  --input-size 224 ^
  --val-ratio 0.2
```

训练脚本会优先按 `source` 分组切分，因此同一张原图裁出的多个 ROI 不会故意同时被分到 train 和 validation。

训练完成后先做诊断：

```bat
python tools\diagnose_screw_classifier.py ^
  --dataset "artifacts\screw_dataset_v2" ^
  --checkpoint "artifacts\screw_classifier_v2\best.pt"
```

再用完全没参与训练的原始图片执行 `inspect_product.py`。
