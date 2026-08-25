#!/usr/bin/env python3
"""
在个人采集的测试集上评估 cup+mouse 检测模型。

测试协议：每个测试图片包含一个已知类别物体。
判定正确 = 检测到物体 且 类别正确 且 置信度 >= --conf。

输出：
1. 按类别分别统计正确率（cup / mouse）
2. 混淆情况（cup / mouse / none）
3. 保存 test_results.csv
4. 把错误案例复制到 results/error_cases/ 供报告使用

用法：
    python3 scripts/evaluate.py \
        --model runs/detect/combined_model_v3/weights/best.pt \
        --test-root test_images \
        --conf 0.25
"""
import argparse
import csv
import os
import shutil
from collections import defaultdict

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='按类别评估 cup+mouse 检测模型')
    ap.add_argument('--model', required=True, help='模型权重路径')
    ap.add_argument('--test-root', default='test_images', help='测试图片根目录（按类别分子目录）')
    ap.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default='results', help='结果输出目录')
    ap.add_argument('--classes', nargs='+', default=['cup', 'mouse'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    err_dir = os.path.join(args.out, 'error_cases')
    os.makedirs(err_dir, exist_ok=True)

    model = YOLO(args.model)
    results_rows = []
    per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
    confusion = defaultdict(int)  # (true, pred) 计数，pred 可为 'none'

    for true_class in args.classes:
        class_dir = os.path.join(args.test_root, true_class)
        if not os.path.isdir(class_dir):
            print(f"警告：{class_dir} 不存在，跳过")
            continue
        for img_file in sorted(os.listdir(class_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            res = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 取该图中置信度最高的框作判定
            pred_class = 'none'
            conf = 0.0
            bbox = []
            if res.boxes is not None and len(res.boxes) > 0:
                idx = res.boxes.conf.argmax()
                conf = float(res.boxes.conf[idx])
                pred_class = res.names[int(res.boxes.cls[idx])]
                bbox = [round(v, 1) for v in res.boxes.xyxy[idx].tolist()]

            correct = (pred_class == true_class)
            per_class[true_class]['total'] += 1
            if correct:
                per_class[true_class]['correct'] += 1
            confusion[(true_class, pred_class)] += 1

            results_rows.append({
                'image': img_file, 'true_class': true_class, 'pred_class': pred_class,
                'confidence': round(conf, 3), 'bbox': bbox, 'correct': correct,
            })

            if not correct:
                shutil.copy2(img_path, os.path.join(
                    err_dir, f'{true_class}_x_{pred_class}_{img_file}'))

    # 保存 CSV
    csv_path = os.path.join(args.out, 'test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'true_class', 'pred_class',
                                               'confidence', 'bbox', 'correct'])
        writer.writeheader()
        writer.writerows(results_rows)

    # 汇总输出
    print(f"=== 评估结果 (conf={args.conf}) ===")
    total = correct_all = 0
    for cls in args.classes:
        st = per_class[cls]
        acc = st['correct'] / st['total'] if st['total'] else 0
        total += st['total']
        correct_all += st['correct']
        print(f"  {cls:6s}: {st['correct']}/{st['total']} = {acc:.1%}")
    print(f"  总体  : {correct_all}/{total} = {correct_all/total:.1%}")

    print("\n=== 混淆情况 (true → pred) ===")
    for (true, pred), n in sorted(confusion.items()):
        print(f"  {true:6s} → {pred:6s} : {n}")

    print(f"\n结果 CSV: {csv_path}")
    print(f"错误案例: {err_dir}")


if __name__ == '__main__':
    main()
