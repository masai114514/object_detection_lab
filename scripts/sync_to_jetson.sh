#!/usr/bin/env bash
# 把训练好的模型和部署代码打包同步到 Jetson Orin NX。
#
# 用法：
#   bash scripts/sync_to_jetson.sh --local [权重路径]         # 只打包到本目录，不传输
#   bash scripts/sync_to_jetson.sh <jetson用户@jetsonIP> [权重路径]
#   bash scripts/sync_to_jetson.sh jetson@192.168.1.100
#   bash scripts/sync_to_jetson.sh jetson@192.168.1.100 runs/detect/combined_model_v3/weights/best.pt
#
# 在 Jetson 上解包后：
#   # 1. 导出 TensorRT engine
#   python3 scripts/export_jetson.py --model models/combined_best.pt --half
#   # 2. 实时识别（有显示 / SSH 无显示加 --no-show）
#   python3 jetson/detect.py --model models/combined_best.engine
#   # 3. 离线复跑 20 张测试集（可选，验证 >=80%）
#   python3 scripts/evaluate.py --model models/combined_best.pt --test-root test_images
#   # 4. ROS2（在 Jetson 上构建并运行）
#   cd ros2_ws && colcon build --symlink-install && source install/setup.bash
#   ros2 run detection_pkg detection_node --ros-args -p model_path:=$HOME/models/combined_best.engine

set -euo pipefail

WEIGHT="runs/detect/combined_model_v3/weights/best.pt"
TARGET=""
if [ "${1:-}" = "--local" ]; then
    WEIGHT="${2:-$WEIGHT}"
elif [ -n "${1:-}" ]; then
    TARGET="$1"
    WEIGHT="${2:-$WEIGHT}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
PKG="$STAGE/jetson_deploy"

mkdir -p "$PKG/models" "$PKG/scripts" "$PKG/jetson"
cp "$ROOT/$WEIGHT" "$PKG/models/combined_best.pt"
cp "$ROOT/scripts/export_jetson.py" "$PKG/scripts/"
cp "$ROOT/scripts/evaluate.py" "$PKG/scripts/"
cp "$ROOT/jetson/detect.py" "$PKG/jetson/"
cp -r "$ROOT/test_images" "$PKG/test_images"
cp -r "$ROOT/ros2_ws" "$PKG/ros2_ws"

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    -C "$STAGE" -czf "$STAGE/jetson_deploy.tar.gz" jetson_deploy
echo "打包完成: $STAGE/jetson_deploy.tar.gz ($(du -h "$STAGE/jetson_deploy.tar.gz" | cut -f1))"

if [ -n "$TARGET" ]; then
    echo "传输到 $TARGET ..."
    scp "$STAGE/jetson_deploy.tar.gz" "$TARGET:~/"
else
    cp "$STAGE/jetson_deploy.tar.gz" "$ROOT/jetson_deploy.tar.gz"
    echo "已复制到: $ROOT/jetson_deploy.tar.gz（用 U 盘/scp 拷到 Jetson）"
fi

echo "在 Jetson 上解包："
echo "  mkdir -p ~/jetson_app && tar -xzf ~/jetson_deploy.tar.gz -C ~/jetson_app"
echo "  然后按 docs/jetson_deploy.md 的四步执行"
rm -rf "$STAGE"
