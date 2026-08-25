#!/usr/bin/env python3
"""
构建平衡的 cup+mouse 合并数据集。

流程：
1. 从 cup 源目录收集杯子样本（YOLO 格式，class 0）
2. 从 mouse 源目录收集鼠标样本（YOLO 格式，class 1）
3. 对多数类（cup）做上限截断，缓解类别不平衡
4. 按类别分层划分 train/val
5. 输出到 {out} 目录，并生成 data_balanced.yaml（相对路径，Mac/Jetson 通用）

用法：
    python3 scripts/build_combined_dataset.py \
        --cup-src public_data/combined \
        --mouse-src public_data/coco/extracted \
        --out public_data/balanced \
        --max-cups 1500 \
        --val-split 0.15
"""
import argparse
import os
import random
import shutil


IMG_EXTS = ('.jpg', '.jpeg', '.png')


def _sibling_labels_dir(img_dir: str) -> str:
    """给定含图片文件的目录，返回对应的标签目录。

    规则：把路径中最后一个 'images' 段替换为 'labels'，兼容三种布局：
    - 扁平：{dir}/images/*.jpg  -> {dir}/labels/
    - Roboflow：{dir}/train/images/*.jpg -> {dir}/train/labels/
    - 拆分：{dir}/images/train/*.jpg -> {dir}/labels/train/
    """
    parts = img_dir.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return os.path.join(*parts)
    # 没有 images 段：假定 labels 与图片同级
    return os.path.join(img_dir, 'labels')


def collect_samples(data_dir: str, class_id: int):
    """
    收集一个 YOLO 数据目录下的所有样本，返回 [(图片路径, 标签文本)]。

    递归扫描所有直接含图片文件的目录，对应标签目录由 _sibling_labels_dir 推导。
    标签中的类别号统一重映射为 class_id（合并时 mouse 源用 1）。
    """
    samples = []
    seen = set()
    for root, _, files in os.walk(data_dir):
        if not any(f.lower().endswith(IMG_EXTS) for f in files):
            continue
        lbl_dir = _sibling_labels_dir(root)
        for fname in sorted(files):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(root, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, f'{stem}.txt')
            if not os.path.exists(lbl_path):
                continue
            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()
            if not lines:
                continue
            # 重映射类别号
            remapped = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = str(class_id)
                    remapped.append(' '.join(parts))
            if not remapped:
                continue
            key = os.path.abspath(img_path)
            if key in seen:
                continue
            seen.add(key)
            samples.append((img_path, '\n'.join(remapped)))
    return samples


def write_sample(out_img_dir, out_lbl_dir, img_path, label_txt, prefix):
    """复制图片并写标签，返回文件名 stem。"""
    stem = f"{prefix}_{os.path.splitext(os.path.basename(img_path))[0]}"
    shutil.copy2(img_path, os.path.join(out_img_dir, f'{stem}.jpg'))
    with open(os.path.join(out_lbl_dir, f'{stem}.txt'), 'w') as f:
        f.write(label_txt + '\n')
    return stem


def main():
    ap = argparse.ArgumentParser(description='构建平衡的 cup+mouse 合并数据集')
    ap.add_argument('--cup-src', required=True, help='杯子源目录（含 images/labels）')
    ap.add_argument('--mouse-src', required=True, help='鼠标源目录（含 images/labels，class 1）')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--max-cups', type=int, default=1500, help='杯子类最大样本数（缓解不平衡）')
    ap.add_argument('--val-split', type=float, default=0.15, help='验证集比例')
    ap.add_argument('--seed', type=int, default=42, help='随机种子，保证可复现')
    args = ap.parse_args()

    random.seed(args.seed)
    cups = collect_samples(args.cup_src, 0)
    mice = collect_samples(args.mouse_src, 1)

    print(f"收集到 cups: {len(cups)} 张, mice: {len(mice)} 张")
    if not mice:
        raise SystemExit("错误：没有收集到任何鼠标样本，请先运行 extract_coco.py 抽取鼠标数据")

    # 平衡：多数类截断
    if len(cups) > args.max_cups:
        random.shuffle(cups)
        cups = cups[:args.max_cups]
        print(f"杯子类超过上限，随机截断到 {args.max_cups} 张")

    # 分层划分（每类独立 85/15）
    def split(items):
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_split))
        return items[n_val:], items[:n_val]

    cup_train, cup_val = split(cups)
    mouse_train, mouse_val = split(mice)
    print(f"划分后: train cups={len(cup_train)} mice={len(mouse_train)} | "
          f"val cups={len(cup_val)} mice={len(mouse_val)}")

    # 写输出：train 分区 = cups + mice，val 分区 = cups + mice
    # 加前缀避免重名（cu_* 杯子 / mo_* 鼠标）
    for split_name in ['train', 'val']:
        out_img = os.path.join(args.out, 'images', split_name)
        out_lbl = os.path.join(args.out, 'labels', split_name)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        cup_items = cup_train if split_name == 'train' else cup_val
        mouse_items = mouse_train if split_name == 'train' else mouse_val
        for img_path, label_txt in cup_items:
            write_sample(out_img, out_lbl, img_path, label_txt, 'cu')
        for img_path, label_txt in mouse_items:
            write_sample(out_img, out_lbl, img_path, label_txt, 'mo')

    # 生成 data_balanced.yaml（相对路径：相对于 yaml 所在目录）
    yaml_path = os.path.join(args.out, 'data_balanced.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# 平衡后的 cup(0)+mouse(1) 数据集\n")
        f.write(f"path: {args.out}\n")  # 与 yaml 同目录
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 2\n")
        f.write(f"names: ['cup', 'mouse']\n")
    print(f"生成数据集配置: {yaml_path}")
    print(f"最终比例 cup:mouse = {len(cup_train) + len(cup_val)}:{len(mouse_train) + len(mouse_val)}"
          f" = {round((len(cup_train)+len(cup_val))/max(1,len(mouse_train)+len(mouse_val)), 2)}:1")


if __name__ == '__main__':
    main()
