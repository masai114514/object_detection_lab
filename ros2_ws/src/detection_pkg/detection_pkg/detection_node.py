#!/usr/bin/env python3
"""
cup+mouse 目标检测 ROS2 发布节点。

功能：
- 摄像头实时检测，把识别结果（类别 / 置信度 / 检测框）发布到 /detections 话题（JSON 字符串）
- 实时画面绘制检测框 + FPS
- 按 's' 保存当前帧并记录到 CSV（手动输入真值类别，统计正确率）
- 错误案例自动存入 results/error_cases/
- 按 'q' 退出

构建（在 Jetson 上，需先 source ROS2 环境）：
    cd ros2_ws && colcon build --symlink-install && source install/setup.bash

运行：
    ros2 run detection_pkg detection_node --ros-args \
        -p model_path:=/home/jetson/models/combined_best.engine \
        -p camera_index:=0
    # 另一终端订阅：ros2 topic echo /detections
"""
import argparse
import csv
import json
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

CLASS_NAMES = ['cup', 'mouse']


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.declare_parameter('model_path',
                               '/home/jetson/models/combined_best.engine')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('topic', 'detections')
        self.declare_parameter('out_dir', 'results')

        model_path = self.get_parameter('model_path').value
        camera_index = self.get_parameter('camera_index').value
        self.conf = self.get_parameter('conf').value
        self.imgsz = self.get_parameter('imgsz').value
        topic = self.get_parameter('topic').value
        out_dir = self.get_parameter('out_dir').value

        self.publisher_ = self.create_publisher(String, topic, 10)
        self.get_logger().info(f'加载模型: {model_path}')
        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'error_cases'), exist_ok=True)
        self.csv_file = open(os.path.join(out_dir, 'test_results.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'true_class', 'pred_class', 'confidence', 'bbox'])

        self.fps = 0.0
        self.last_time = time.time()

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
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
                    cv2.putText(frame, f'{cls_name} {conf:.2f}',
                                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self.last_time + 1e-6))
            self.last_time = now
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 发布识别结果
            msg = String()
            msg.data = json.dumps(detections)
            self.publisher_.publish(msg)

            cv2.imshow('cup+mouse detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'results/{ts}.jpg', frame)
                true_cls = input('真值类别 (0=cup, 1=mouse): ').strip()
                true_name = CLASS_NAMES[0] if true_cls == '0' else CLASS_NAMES[1]
                pred = detections[0] if detections else {'class': 'none', 'confidence': 0, 'bbox': []}
                self.csv_writer.writerow([ts, true_name, pred['class'],
                                          pred['confidence'], json.dumps(pred['bbox'])])
                self.csv_file.flush()
                if true_name != pred['class']:
                    cv2.imwrite(f'results/error_cases/{ts}_{true_name}_{pred["class"]}.jpg', frame)
                    self.get_logger().warn('错误案例已保存')
            elif key == ord('q'):
                break

        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()

    def destroy_node(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
