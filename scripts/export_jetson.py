#!/usr/bin/env python3
"""
在 Jetson (Orin NX) 上把训练好的 best.pt 转成 TensorRT engine，用于 5 FPS+ 实时推理。

注意：
- 必须在 Jetson 上运行（需要 CUDA + TensorRT），Mac 上只能导出 ONNX。
- 转换后得到 {best_name}.engine，推理加载它会快很多（Orin NX 上 yolov8s@640 可达 30+ FPS）。

在 Jetson 上运行：
    python3 scripts/export_jetson.py --model best.pt --half

等价于 yolo CLI：
    yolo export model=best.pt format=engine half=True imgsz=640 device=0
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='导出 TensorRT engine（在 Jetson 上运行）')
    ap.add_argument('--model', required=True, help='best.pt 路径')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--half', action='store_true', help='FP16 量化，显著提速（推荐）')
    args = ap.parse_args()

    model = YOLO(args.model)
    out = model.export(
        format='engine',
        imgsz=args.imgsz,
        half=args.half,
        device=0,          # Jetson CUDA
    )
    print(f"导出完成: {out}")


if __name__ == '__main__':
    main()
