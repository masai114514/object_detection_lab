#!/usr/bin/env python3
"""
采集个人数据（cup / mouse）。

两种用途（--mode）：
  train  -> 保存到 personal_data/{class}/   用于补充训练集（需自行标注，用 Roboflow / labelImg）
  test   -> 保存到 test_images/{class}/     用于最终测试（不参与训练）

建议采集时变化：不同角度、距离、光照、遮挡、背景，提高泛化能力。

用法：
    python3 scripts/collect_data.py --object cup --mode train
    python3 scripts/collect_data.py --object mouse --mode test

按 's' 保存，按 'q' 退出。
"""
import argparse
import os
from datetime import datetime

import cv2


def main():
    ap = argparse.ArgumentParser(description='采集个人 cup/mouse 数据')
    ap.add_argument('--object', choices=['cup', 'mouse'], required=True)
    ap.add_argument('--mode', choices=['train', 'test'], default='test')
    ap.add_argument('--source', default='0')
    args = ap.parse_args()

    base = 'personal_data' if args.mode == 'train' else 'test_images'
    save_dir = os.path.join(base, args.object)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    print(f"采集 {args.object} 类({args.mode})，保存到 {save_dir}/，按 's' 保存，'q' 退出")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('collect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            path = os.path.join(save_dir, f'{ts}.jpg')
            cv2.imwrite(path, frame)
            count += 1
            print(f'已保存 {count}: {path}')
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"共采集 {count} 张 -> {save_dir}/")


if __name__ == '__main__':
    main()
