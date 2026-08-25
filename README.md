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
│   ├── mouse_rf_653/           # Roboflow Computer-Mouse v2（653 项目、1110 张，CC BY 4.0）
│   ├── mouse_rf/               # Roboflow Computer-Mouse v15（5747 张，可选补充）
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
# 1.1 获取鼠标样本（Roboflow Computer-Mouse，CC BY 4.0）
# 用 scripts/download_dataset.py 里的 Roboflow API key，通过 rfapi.get_version_export()
# 拿到直链后用 aria2 下载（GitHub/部分网站在国内被限速，见「已知问题」）：
#   项目 machine-learning-chipg/computer-mouse-tqzgh v2  → public_data/mouse_rf_653/
#   项目 (Computer-Mouse v15, 5747 张)                   → public_data/mouse_rf/（可选补充）
# 解压后得到 {train,valid,test}/images + {train,valid,test}/labels（black-mouse + white-mouse 两类）
# 构建脚本会把它们统一重映射为 class 1 (mouse)。

# 1.2 合并 cup + mouse，平衡多数类，划分 train/val
# 输出 1500 cup : 1110 mouse（实例级 2310:1118），无 train/val 交叉重复
python3 scripts/build_combined_dataset.py \
    --cup-src public_data/combined \
    --mouse-src public_data/mouse_rf_653/extracted \
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

- 旧合并数据集 `public_data/combined` **缺少鼠标标注**（3191 cup / 0 mouse），导致旧模型把鼠标全识别成杯子（测试 10 只鼠标 0 命中）。
  已改用 Roboflow Computer-Mouse 数据集重建平衡数据集（1500 cup : 1110 mouse）。
- 最初尝试从 COCO val2017 抽取鼠标数据，但 COCO 官方下载在国内被严重限速且标注 zip CRC 损坏，故弃用 COCO、改用 Roboflow API。
- **GitHub Releases 在本机不可达**（github.com 下载超时），ultralytics 自动下载权重会卡死；
  `train_combined.py` 已加 `ensure_weights()`：本地无权重时从 hf-mirror.com 镜像下载。
- `build_combined_dataset.py` 已修复两处 bug：①支持扁平 / Roboflow / 拆分三种目录布局；
  ②重建前清空输出目录，避免上一次运行残留导致 train/val 交叉重复（数据泄漏）。
- 公开数据训练的模型对个人桌面环境存在域差异，建议用 `collect_data.py --mode train` 采集
  100+ 张个人图片（含鼠标）补充训练，或用 `--max-cups` 加入更多个人杯数据。
- `data_combined.yaml` 中的 `/root` 路径为 Jetson 专用；Mac 训练请使用 `public_data/balanced/data_balanced.yaml`。
