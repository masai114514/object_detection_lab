#!/usr/bin/env bash
# AutoDL (RTX 5090 / 任何 CUDA GPU) 上的一键训练脚本。
#
# 用法（在 AutoDL 实例上）：
#   cd ~/autodl-tmp/object_detection_lab
#   bash cloud/run_training.sh
#
# 训练结束后 best.pt 在 runs/detect/combined_model_v3/weights/best.pt，
# 并用 evaluate.py 直接给出个人测试集分准确率。

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. 环境检查 ==="
python3 -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python3 -c "import ultralytics" 2>/dev/null || { echo "安装 ultralytics ..."; pip install -q -U ultralytics; }

echo "=== 2. 开始训练（yolov8s, 150 epochs, CUDA）==="
python3 scripts/train_combined.py

echo "=== 3. 在个人测试集上评估 ==="
BEST=runs/detect/combined_model_v3/weights/best.pt
python3 scripts/evaluate.py --model "$BEST" --test-root test_images --conf 0.25

echo ""
echo "=== 完成 ==="
echo "权重: $(pwd)/$BEST"
echo "结果: $(pwd)/results/test_results.csv"
echo "把权重 scp 回本地："
echo "  scp -P <端口> root@<host>:~/autodl-tmp/object_detection_lab/runs/detect/combined_model_v3/weights/best.pt ."
