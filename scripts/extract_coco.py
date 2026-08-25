#!/usr/bin/env python3
"""
从 COCO 数据集中抽取指定类别，转换为 YOLO 标注格式。

用途：补充少数类训练样本（如 mouse）。不需要 pycocotools，纯标准库实现。

用法示例：
    python3 scripts/extract_coco.py \
        --annotations public_data/coco/raw/annotations/instances_val2017.json \
        --images public_data/coco/raw/val2017 \
        --out public_data/coco/extracted \
        --categories mouse:1 cup:0 \
        --min-side 20 --min-area 400

说明：
    --categories 参数格式为 "类别名:输出class id"，多个用空格分隔。
    输出目录结构（YOLO 格式，flat）：
        {out}/images/*.jpg
        {out}/labels/*.txt
    抽取的图片加类别前缀避免重名。
"""
import argparse
import json
import os
import shutil


COCO_ANIMAL_KEYS = {  # 防止误抽成"动物老鼠"，只看目标类别名
}


def load_annotations(annot_path: str) -> dict:
    with open(annot_path) as f:
        return json.load(f)


def build_index(ann_data: dict):
    """返回 (类别名->id 映射, 图片信息, 每个类别的标注列表)。"""
    cat_id2name = {c['id']: c['name'] for c in ann_data['categories']}
    images = {img['id']: img for img in ann_data['images']}
    # 按 image_id 聚合标注，只保留需要解析的字段
    ann_by_image = {}
    for ann in ann_data['annotations']:
        ann_by_image.setdefault(ann['image_id'], []).append(ann)
    return cat_id2name, images, ann_by_image


def main():
    ap = argparse.ArgumentParser(description='从 COCO 抽取指定类别转 YOLO 格式')
    ap.add_argument('--annotations', required=True, help='COCO instances_val2017.json 路径')
    ap.add_argument('--images', required=True, help='COCO val2017 图片目录')
    ap.add_argument('--out', required=True, help='输出目录（YOLO flat 格式）')
    ap.add_argument('--categories', nargs='+', required=True,
                    help='要抽取的类别，格式 "类别名:class_id"，如 mouse:1')
    ap.add_argument('--min-side', type=float, default=20.0, help='过滤掉边长小于该值的框')
    ap.add_argument('--min-area', type=float, default=400.0, help='过滤掉面积小于该值的框')
    args = ap.parse_args()

    # 解析类别映射
    want = {}
    for item in args.categories:
        name, cls = item.split(':')
        want[name] = int(cls)

    out_img = os.path.join(args.out, 'images')
    out_lbl = os.path.join(args.out, 'labels')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    cat_id2name, images, ann_by_image = build_index(load_annotations(args.annotations))
    wanted_cat_ids = {cid for cid, name in cat_id2name.items() if name in want}

    stats = {name: {'images': 0, 'instances': 0} for name in want}
    skipped = {'too_small': 0, 'no_valid': 0, 'missing_image': 0}

    for img_id, img in images.items():
        anns = ann_by_image.get(img_id, [])
        # 该图片里我们关心的标注
        hits = [a for a in anns if a['category_id'] in wanted_cat_ids]
        if not hits:
            continue
        w, h = img['width'], img['height']
        if not w or not h:
            continue

        # 按类别分组，分别写 label；一张图可能同时有 cup 和 mouse
        valid_lines = []       # (输出类名, yolo 文本)
        valid_cats = set()
        for a in hits:
            x, y, bw, bh = a['bbox']
            cls_name = cat_id2name[a['category_id']]
            out_cls = want[cls_name]
            if bw < args.min_side or bh < args.min_side or bw * bh < args.min_area:
                skipped['too_small'] += 1
                continue
            cx, cy = x + bw / 2, y + bh / 2
            # 归一化并截断到 [0,1]
            nx, ny = min(cx / w, 1.0), min(cy / h, 1.0)
            nw, nh = min(bw / w, 1.0), min(bh / h, 1.0)
            valid_lines.append(f"{out_cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            valid_cats.add(cls_name)
        if not valid_lines:
            skipped['no_valid'] += 1
            continue

        src = os.path.join(args.images, img['file_name'])
        if not os.path.exists(src):
            skipped['missing_image'] += 1
            continue

        # 前缀：类别名_原文件名，避免合并数据集时重名
        stem = os.path.splitext(img['file_name'])[0]
        prefix = '_'.join(sorted(valid_cats))
        dst_img = os.path.join(out_img, f"{prefix}_{stem}.jpg")
        shutil.copy2(src, dst_img)
        with open(os.path.join(out_lbl, f"{prefix}_{stem}.txt"), 'w') as f:
            f.write('\n'.join(valid_lines) + '\n')

        for c in valid_cats:
            stats[c]['images'] += 1
            stats[c]['instances'] += len([v for v in valid_lines if int(v.split()[0]) == want[c]])

    print("=== 抽取结果 ===")
    for name, s in stats.items():
        print(f"{name}: {s['images']} 张图片, {s['instances']} 个实例 (输出 class {want[name]})")
    print(f"过滤掉的过小框: {skipped['too_small']}, 无有效框的图: {skipped['no_valid']}, 缺图: {skipped['missing_image']}")
    print(f"输出目录: {args.out}")


if __name__ == '__main__':
    main()
