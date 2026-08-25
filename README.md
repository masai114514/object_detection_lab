# Object Detection Lab — cup + mouse 双类目标检测

课程实验一：目标检测与识别。使用 YOLOv8 同时识别桌面上的**杯子**（class 0）和**鼠标**（class 1），在 Jetson Orin NX 上实时运行并通过 ROS2 发布识别结果。

## 环境

- 训练：Mac（Apple Silicon，MPS 加速），`yolov8_env/` 虚拟环境，`ultralytics` + `torch`
- 部署：Jetson Orin NX（TensorRT engine，≥5 FPS）
- 数据：公开数据集（Roboflow/COCO）+ 个人采集数据

## 目录结构

```
object_detection_lab/
├── data_combined.yaml          # 旧版配置（指向 Jetson /root，勿用于 Mac）
├── public_data/
│   ├── cup/                    # 公开杯子数据集（Roboflow，class 0）
│   ├── combined/               # 旧合并数据集（缺鼠标标注，已弃用）
│   ├── coco/
│   │   ├── raw/                # COCO 下载缓存（val2017.zip + 标注）
│   │   └── extracted/          # 从 COCO 抽取的鼠标图片（class 1）
│   └── balanced/               # 平衡后的合并数据集（train/val + data_balanced.yaml）
├── personal_data/              # 个人采集训练数据（可选，用于减少域差异）
├── test_images/{cup,mouse}     # 个人测试集（20 张，10+10，不参与训练）
├── scripts/
│   ├── extract_coco.py         # 从 COCO 抽取 mouse/cup，转 YOLO 格式
│   ├── build_combined_dataset.py # 合并 cup+mouse、平衡、划分 train/val
│   ├── train_combined.py       # 训练双类模型（yolov8s, cos_lr, MPS/CUDA 自动）
│   ├── evaluate.py             # 按类别评估测试集，保存结果与错误案例
│   ├── export_jetson.py        # 在 Jetson 上导出 TensorRT engine
│   ├── collect_data.py         # 采集个人数据（--mode train/test）
│   └── download_dataset.py     # 从 Roboflow 下载公开数据集
├── jetson/
│   └── detect.py               # Jetson 实时识别程序（FPS + 保存结果）
└── ros2_ws/
    └── src/detection_pkg/      # ROS2 发布节点（detections 话题）
```

## 完整流程

### 1. 准备数据集

```bash
# 1.1 从 COCO 抽取鼠标样本（补少数类）——需先下载 COCO val2017（见 public_data/coco/raw/）
python3 scripts/extract_coco.py \
    --annotations public_data/coco/raw/annotations/instances_val2017.json \
    --images public_data/coco/raw/val2017 \
    --out public_data/coco/extracted \
    --categories mouse:1 \
    --min-side 20 --min-area 400

# 1.2 合并 cup + mouse，平衡多数类，划分 train/val
python3 scripts/build_combined_dataset.py \
    --cup-src public_data/combined \
    --mouse-src public_data/coco/extracted \
    --out public_data/balanced \
    --max-cups 1500 \
    --val-split 0.15
```

### 2. 训练（Mac 上用 MPS）

```bash
python3 scripts/train_combined.py                 # yolov8s, 150 epochs, cos_lr
python3 scripts/train_combined.py --model yolov8m.pt --epochs 200   # 可选调参
```

权重输出到 `runs/detect/combined_model_v3/weights/best.pt`。训练完成后查看 `results.csv` 中
`metrics/mAP50(B)`，双类任务目标 ≥ 0.85。

### 3. 评估（个人测试集）

```bash
python3 scripts/evaluate.py \
    --model runs/detect/combined_model_v3/weights/best.pt \
    --test-root test_images \
    --conf 0.25
```

按类别分别统计准确率（验收要求：整体 ≥80%，建议 cup/mouse 各自 ≥80%），
结果存 `results/test_results.csv`，错误案例自动复制到 `results/error_cases/`。

### 4. Jetson 部署

```bash
# 4.1 把 best.pt 拷到 Jetson，导出 TensorRT engine（在 Jetson 上运行）
python3 scripts/export_jetson.py --model best.pt --half

# 4.2 实时识别（独立程序）
python3 jetson/detect.py --model /home/jetson/models/combined_best.engine

# 4.3 ROS2 发布（识别结果 → /detections 话题）
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 run detection_pkg detection_node --ros-args \
    -p model_path:=/home/jetson/models/combined_best.engine
```

`jetson/detect.py` 按 `s` 保存帧并记录真值（0=cup, 1=mouse），自动统计正确率并保存错误案例。

## 验收要点对照

| 要求 | 对应实现 |
|------|----------|
| 同时识别 ≥2 类物体 | cup + mouse，`nc=2` |
| 20 个物体正确率 ≥80% | `scripts/evaluate.py` 按类统计 |
| Jetson ≥5 FPS | TensorRT FP16，yolov8s@640 实测 30+ FPS |
| 保存结果和错误案例 | `results/test_results.csv` + `results/error_cases/` |
| ROS2 发布识别结果 | `detection_node` 发布 `/detections` |

## 已知问题与改进记录

- 旧合并数据集 `public_data/combined` **缺少鼠标标注**（3191 cup / 0 mouse），导致旧模型把鼠标全识别成杯子。已通过 COCO 抽取鼠标样本重建平衡数据集。
- 公开数据训练的模型对个人桌面环境存在域差异，建议用 `collect_data.py --mode train` 采集
  100+ 张个人图片（含鼠标）补充训练。
- `data_combined.yaml` 中的 `/root` 路径为 Jetson 专用；Mac 训练请使用 `public_data/balanced/data_balanced.yaml`。
