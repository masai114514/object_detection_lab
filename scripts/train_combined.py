#!/usr/bin/env python3
"""
训练 cup+mouse 双类目标检测模型。

相对旧版 train_combined.py 的改进：
1. 模型从 yolov8n 升级为 yolov8s（Orin NX / Mac 都轻松跑）
2. lr0=0.001 -> 0.01（预训练迁移学习的标准初始学习率）
3. 增加 cos_lr 余弦退火 + 更长训练周期（150 epoch）
4. 设备自动检测：Jetson 用 CUDA，Mac 用 MPS
5. 数据集用平衡后的 data_balanced.yaml（旧版 data_combined.yaml 指向 Jetson 路径，Mac 上无法运行）

用法：
    python3 scripts/train_combined.py                 # 全部用默认值
    python3 scripts/train_combined.py --model yolov8m.pt --epochs 200 --batch 16
"""
import argparse
import torch
from ultralytics import YOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return '0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser(description='训练 cup+mouse 双类检测模型')
    ap.add_argument('--data', default='public_data/balanced/data_balanced.yaml',
                    help='数据集配置（平衡后的）')
    ap.add_argument('--model', default='yolov8s.pt', help='预训练权重')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr0', type=float, default=0.01)
    ap.add_argument('--name', default='combined_model_v3')
    ap.add_argument('--workers', type=int, default=0,
                    help='数据加载线程数；Mac 上 MPS 建议 0，Jetson 可设 4~8')
    args = ap.parse_args()

    device = pick_device()
    print(f"使用设备: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        cos_lr=True,
        patience=30,
        workers=args.workers,
        name=args.name,
    )


if __name__ == '__main__':
    main()
