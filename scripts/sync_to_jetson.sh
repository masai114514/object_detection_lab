#!/usr/bin/env bash
# 把训练好的模型和部署代码打包同步到 Jetson Orin NX。
#
# 用法：
#   bash scripts/sync_to_jetson.sh <jetson用户@jetsonIP> [模型权重]
#   bash scripts/sync_to_jetson.sh jetson@192.168.1.100
#   bash scripts/sync_to_jetson.sh jetson@192.168.1.100 runs/detect/combined_model_v3/weights/best.pt
#
# 在 Jetson 上解包后：
#   # 1. 导出 TensorRT engine
#   python3 scripts/export_jetson.py --model models/combined_best.pt --half
#   # 2. 实时识别
#   python3 jetson/detect.py --model models/combined_best.engine
#   # 3. ROS2（在 Jetson 上构建并运行）
#   cd ros2_ws && colcon build --symlink-install && source install/setup.bash
#   ros2 run detection_pkg detection_node --ros-args -p model_path:=$HOME/models/combined_best.engine

set -euo pipefail

TARGET="${1:?用法: $0 <jetson用户@IP> [权重路径]}"
WEIGHT="${2:-runs/detect/combined_model_v3/weights/best.pt}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
PKG="$STAGE/jetson_deploy"

mkdir -p "$PKG/models" "$PKG/scripts" "$PKG/jetson"
cp "$ROOT/$WEIGHT" "$PKG/models/combined_best.pt"
cp "$ROOT/scripts/export_jetson.py" "$PKG/scripts/"
cp "$ROOT/jetson/detect.py" "$PKG/jetson/"
cp -r "$ROOT/ros2_ws" "$PKG/ros2_ws"

tar -C "$STAGE" -czf "$STAGE/jetson_deploy.tar.gz" jetson_deploy
echo "打包完成: $STAGE/jetson_deploy.tar.gz ($(du -h "$STAGE/jetson_deploy.tar.gz" | cut -f1))"
echo "传输到 $TARGET ..."
scp "$STAGE/jetson_deploy.tar.gz" "$TARGET:~/"
echo "在 Jetson 上解包："
echo "  mkdir -p ~/jetson_app && tar -xzf ~/jetson_deploy.tar.gz -C ~/jetson_app"
echo "  然后按文件头注释的三步执行"
rm -rf "$STAGE"
