#!/usr/bin/env python3
"""
从 Roboflow 下载公开数据集（cup + mouse），供 build_combined_dataset.py 使用。

用法：
    export ROBOFLOW_API_KEY="<你的 Roboflow API key>"   # 官网 My Settings → API Key
    python3 scripts/download_dataset.py                      # 下载 cup v3 + mouse v2
    python3 scripts/download_dataset.py --include v15        # 额外下载 mouse v15（补充）

API key 从环境变量读取，不硬编码在代码里（避免提交到 GitHub 泄露）。

下载源（教师要求提供的原始数据链接）：
    cup  :  Roboflow cup-detection v3（CC BY 4.0）
            https://universe.roboflow.com/my-workspace-7j2fi/cup-detection-w8kfb/dataset/3
    mouse:  Roboflow Computer-Mouse v2（CC BY 4.0，653 项目 / 1110 张）
            https://universe.roboflow.com/machine-learning-chipg/computer-mouse-tqzgh/dataset/2
    mouse 补充: Computer-Mouse v15（5747 张，可选）
            https://universe.roboflow.com/project-e9bly/computer-mouse-fmjvk/dataset/15

说明：国内网络从 Roboflow 下载慢时，可在浏览器打开上述页面点 Download → 复制
zip 直链用 aria2 加速：aria2c -x 8 -s 8 "<zip-url>"（README 1.1 节有详述）。
"""
import argparse
import os
import sys

from roboflow import Roboflow

# (名称, workspace, project, version, 输出目录)
DOWNLOADS = [
    ('cup-detection v3',       'my-workspace-7j2fi',   'cup-detection-w8kfb',   3, 'public_data/cup'),
    ('computer-mouse v2',      'machine-learning-chipg', 'computer-mouse-tqzgh', 2, 'public_data/mouse_rf_653'),
    ('computer-mouse v15',     'project-e9bly',        'computer-mouse-fmjvk', 15, 'public_data/mouse_rf'),
]


def main():
    ap = argparse.ArgumentParser(description='下载 Roboflow cup/mouse 公开数据集')
    ap.add_argument('--include', default='v2',
                    help="下载范围：'v2'（默认，cup v3 + mouse v2）或 'v15'（额外加 mouse v15）")
    args = ap.parse_args()

    key = os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        sys.exit('请先设置环境变量 ROBOFLOW_API_KEY（Roboflow 官网获取）')

    targets = DOWNLOADS
    if args.include != 'v15':
        targets = DOWNLOADS[:2]
    print(f'将下载 {len(targets)} 个数据集到 public_data/')
    rf = Roboflow(api_key=key)
    for name, ws, proj, ver, out in targets:
        print(f'[1/1] 下载 {name} -> {out}/')
        project = rf.workspace(ws).project(proj)
        project.version(ver).download('yolov8', location=out)
    print('全部完成。下一步：python3 scripts/build_combined_dataset.py')


if __name__ == '__main__':
    main()
