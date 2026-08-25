#!/usr/bin/env bash
# 生成 AutoDL 云训练包 cloud_train.tar.gz（数据集 + 脚本 + 权重 + 个人测试集）。
#
# 用法：
#   bash scripts/sync_cloud.sh              # 打包到 ./cloud_train.tar.gz
#   bash scripts/sync_cloud.sh <目标路径>   # 打包到指定路径
#
# 上传（AutoDL 实例的 SSH 端口/地址在控制台查看）：
#   scp -P <端口> cloud_train.tar.gz root@<地址>:/root/autodl-tmp/
# 在实例上：
#   cd ~/autodl-tmp && tar -xzf cloud_train.tar.gz && bash cloud/run_training.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/cloud_train.tar.gz}"
cd "$ROOT"

# 重新生成云训练包时确保最新脚本
tar --exclude='runs' \
    --exclude='public_data/mouse_rf' \
    --exclude='public_data/mouse_rf_653' \
    --exclude='public_data/coco' \
    --exclude='public_data/cup' \
    --exclude='public_data/combined' \
    --exclude='.git' --exclude='.DS_Store' --exclude='__pycache__' \
    -czf "$OUT" \
    public_data/balanced scripts cloud yolov8s.pt test_images README.md

echo "云训练包已生成: $OUT ($(du -h "$OUT" | cut -f1))"
