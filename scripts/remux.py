#!/usr/bin/env python3
"""按目标 fps 重封装视频（帧内容不变，仅改播放速度/时长）。用法: remux.py <src> <dst> <fps>"""
import sys
import cv2

src, dst, fps_out = sys.argv[1], sys.argv[2], float(sys.argv[3])
cap = cv2.VideoCapture(src)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (w, h))
cnt = 0
while True:
    ret, f = cap.read()
    if not ret:
        break
    vw.write(f)
    cnt += 1
vw.release()
cap.release()
print(f"remux 完成: {cnt} 帧 -> {fps_out:.3f} fps, 播放时长 {cnt / fps_out:.1f}s")
