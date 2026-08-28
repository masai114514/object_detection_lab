#!/usr/bin/env python3
"""分析录制视频：逐秒统计 杯/鼠 框与 FPS 叠加，确认视频内容符合要求。"""
import sys
import cv2

path = sys.argv[1] if len(sys.argv) > 1 else 'demo_detect.mp4'
cap = cv2.VideoCapture(path)
fps_cap = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur = n / fps_cap if fps_cap > 0 else 0
print(f"文件: {path}\n分辨率: {w}x{h}  fps: {fps_cap:.1f}  帧数: {n}  时长: {dur:.1f}s")

# 每秒采样一帧（取每秒第 0 帧）
sec_prev = -1
summary = []
while True:
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        break
    sec = int(idx / fps_cap) if fps_cap > 0 else idx
    if sec == sec_prev:
        continue
    sec_prev = sec
    # 绿框=cup(0,255,0)  橙框=mouse(255,128,0)  红字FPS(0,0,255) 允许±60容差
    g = cv2.inRange(frame, (0, 195, 0), (60, 255, 60)).sum() // 255
    o = cv2.inRange(frame, (195, 68, 0), (255, 188, 60)).sum() // 255
    r = cv2.inRange(frame[:, :200, :], (0, 0, 195), (60, 60, 255)).sum() // 255
    has_cup = g > 80        # 绿色像素数
    has_mouse = o > 80
    has_fps = r > 20        # 左上角红字
    summary.append((sec, has_cup, has_mouse, has_fps))

# 压缩输出：每 5 秒一段
print(f"\n{'秒':>4} {'杯':>4} {'鼠':>4} {'FPS字':>5}")
for sec, c, m, f_ in summary:
    mark = ''
    if c and m:
        mark = '  <== 同框'
    elif c:
        mark = '  (cup)'
    elif m:
        mark = '  (mouse)'
    print(f"{sec:>4} {('Y' if c else '.'):>4} {('Y' if m else '.'):>4} {('Y' if f_ else '.'):>5}{mark}")

n_cup = sum(1 for _, c, _, _ in summary if c)
n_mouse = sum(1 for _, _, m, _ in summary if m)
n_both = sum(1 for _, c, m, _ in summary if c and m)
n_fps = sum(1 for _, _, _, f_ in summary if f_)
print(f"\n统计: 杯框 {n_cup}s, 鼠框 {n_mouse}s, 同框 {n_both}s, FPS字 {n_fps}s")
cap.release()
