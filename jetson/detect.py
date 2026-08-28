#!/usr/bin/env python3
"""
Jetson 实时目标检测程序（cup + mouse）。

功能：
- 摄像头实时识别，绘制目标类别 / 检测框 / 置信度
- 实时显示 FPS（验收要求 >= 5 FPS）
- 按 's' 保存当前帧 + 记录结果到 CSV（含人工标注真值，用于正确率统计）
- 记录错误案例图片到 results/error_cases/
- 按 'q' 退出

两种运行方式：
- 有显示器（或 X11 转发）：默认弹窗预览，按 's'/'q' 交互
- 无显示器（SSH 终端）：加 --no-show，自动周期存帧 + 打印 FPS，不弹窗

在 Jetson 上运行：
    python3 jetson/detect.py --model /home/jetson/models/combined_best.engine            # 有显示器
    python3 jetson/detect.py --model ...engine --no-show --save-interval 3             # SSH 无显示
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='Jetson 实时 cup+mouse 检测')
    ap.add_argument('--model', required=True, help='模型路径（.engine 或 .pt）')
    ap.add_argument('--source', default='0', help='摄像头索引或视频文件')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help='推理设备（.pt 用，默认 GPU 0）')
    ap.add_argument('--half', action='store_true', help='FP16 推理（.pt 用，可提升 FPS）')
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-show', action='store_true',
                    help='无显示环境（SSH 无 DISPLAY）：自动存帧+打印 FPS，不弹窗、不做交互')
    ap.add_argument('--save-interval', type=float, default=3.0,
                    help='--no-show 时每隔多少秒存一张带框帧')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    err_dir = os.path.join(args.out, 'error_cases')
    os.makedirs(err_dir, exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    csv_path = os.path.join(args.out, 'test_results.csv')
    csv_handle = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_handle)
    writer.writerow(['timestamp', 'true_class', 'pred_class', 'confidence', 'bbox'])

    # 显示可用性：显式 --no-show，或在 Linux 下无 DISPLAY（Jetson SSH 场景）
    show = (not args.no_show) and (os.name == 'nt' or sys.platform == 'darwin'
                                   or bool(os.environ.get('DISPLAY')))
    if not show:
        print(f"[headless] 无显示环境，自动每 {args.save_interval:.0f}s 存帧到 {args.out}/，Ctrl-C 退出。")

    fps = 0.0
    last = time.time()
    last_save = 0.0
    print(f"{'按 s 保存+记录 / q 退出。' if show else ''}模型: {args.model}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf,
                        device=args.device, half=args.half, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_name = results.names[int(box.cls[0])]
                detections.append({
                    'class': cls_name, 'confidence': round(conf, 3),
                    'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                })
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0) if cls_name == 'cup' else (255, 128, 0), 2)
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS 显示
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last + 1e-6))
        last = now

        if show:
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('cup+mouse detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                img_path = os.path.join(args.out, f'{ts}.jpg')
                cv2.imwrite(img_path, frame)
                true_cls = input('真值类别 (0=cup, 1=mouse): ').strip()
                true_name = 'cup' if true_cls == '0' else 'mouse'
                pred = detections[0] if detections else {'class': 'none', 'confidence': 0, 'bbox': []}
                writer.writerow([ts, true_name, pred['class'], pred['confidence'],
                                 json.dumps(pred['bbox'])])
                csv_handle.flush()
                if true_name != pred['class']:
                    cv2.imwrite(os.path.join(err_dir, f'{ts}_{true_name}_{pred["class"]}.jpg'), frame)
                    print(f'错误案例已保存: {ts}')
            elif key == ord('q'):
                break
        else:
            # headless：周期打印 FPS 摘要 + 存带框帧
            if now - last_save >= args.save_interval:
                last_save = now
                ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(os.path.join(args.out, f'auto_{ts}.jpg'), frame)
                desc = '; '.join(f"{d['class']} {d['confidence']}" for d in detections) or '无检测'
                print(f"[{ts}] FPS {fps:.1f} | {desc}")

    cap.release()
    csv_handle.close()
    if show:
        cv2.destroyAllWindows()
    print(f"结果已保存到 {args.out}/")


if __name__ == '__main__':
    main()
