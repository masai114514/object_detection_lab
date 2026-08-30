# 数据集说明（balanced 平衡训练集）

本实验的训练数据集位于 `balanced/`，由公开数据（Roboflow）构建，共 **2610 张**
（1500 张 cup : 1110 张 mouse，实例级 3428：cup 2310 / mouse 1118），
按 85:15 分层划分为 `images/train`（2219 张）与 `images/val`（391 张），
YOLO 格式，类别：0 = cup（杯子），1 = mouse（鼠标）。

## 公开数据来源链接（原始数据合计约 2.6GB）

| 数据 | 来源 | 链接 |
|------|------|------|
| 杯子 class 0 | Roboflow `cup-detection` v3（CC BY 4.0），抽取 1500 张 | https://universe.roboflow.com/my-workspace-7j2fi/cup-detection-w8kfb/dataset/3 |
| 鼠标 class 1（主） | Roboflow `Computer-Mouse` v2（CC BY 4.0，653 项目 / 1110 张） | https://universe.roboflow.com/machine-learning-chipg/computer-mouse-tqzgh/dataset/2 |
| 鼠标 class 1（补充，可选） | Roboflow `Computer-Mouse` v15（5747 张） | https://universe.roboflow.com/project-e9bly/computer-mouse-fmjvk/dataset/15 |
| 备选通用数据 | Microsoft COCO 数据集 | https://cocodataset.org/ |

原始 2.6GB 图片因 GitHub 文件大小限制未入库，下载与构建方式：
`scripts/download_dataset.py`（Roboflow API 直链 + aria2）→ `scripts/build_combined_dataset.py`。

## 目录结构

```
public_data/balanced/
├── data_balanced.yaml     # 训练配置文件（相对路径，直接可用）
├── images/
│   ├── train/  (2219 张)
│   └── val/    (391 张)
└── labels/
    ├── train/  (2219 个 txt)
    └── val/    (391 个 txt)
```

构建命令（复现）：
```bash
python3 scripts/build_combined_dataset.py \
    --cup-src public_data/cup \
    --mouse-src public_data/mouse_rf_653/extracted \
    --out public_data/balanced --max-cups 1500 --val-split 0.15
```

> 注：`public_data/cup` 与 `public_data/mouse_rf_653` 为 Roboflow 解压后的原始图片，
> 体积大，同样未入库（GitHub LFS 外文件大小限制）；balanced 产物已完整入库。
