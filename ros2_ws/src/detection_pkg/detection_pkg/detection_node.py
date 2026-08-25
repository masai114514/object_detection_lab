import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import json
import time
import csv
import os

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.publisher_ = self.create_publisher(String, 'detections', 10)
        # 模型路径需根据 Jetson 上实际位置修改
        self.model = YOLO('/home/jetson/models/combined_best.engine')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/error_cases', exist_ok=True)
        self.csv_file = open('results/test_results.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'true_class', 'pred_class', 'confidence', 'bbox'])
        self.fps = 0
        self.last_time = time.time()

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret: break
            results = self.model(frame, imgsz=640, verbose=False)[0]
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = results.names[cls_id]
                    detections.append({'class': cls_name, 'confidence': round(conf,3),
                                       'bbox': [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]})
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    label = f'{cls_name} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            now = time.time()
            self.fps = 1.0 / (now - self.last_time + 1e-6)
            self.last_time = now
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            msg = String()
            msg.data = json.dumps(detections)
            self.publisher_.publish(msg)

            cv2.imshow('Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                img_path = f'results/{timestamp}.jpg'
                cv2.imwrite(img_path, frame)
                true_class = input('Enter true class (0: cup, 1: mouse): ')
                true_name = ['cup','mouse'][int(true_class)]
                pred_name = detections[0]['class'] if detections else 'none'
                conf = detections[0]['confidence'] if detections else 0
                bbox = detections[0]['bbox'] if detections else []
                self.csv_writer.writerow([timestamp, true_name, pred_name, conf, bbox])
                self.csv_file.flush()
                if true_name != pred_name:
                    cv2.imwrite(f'results/error_cases/{timestamp}_{true_name}_{pred_name}.jpg', frame)
                    self.get_logger().warn(f'Error case saved')
            elif key == ord('q'):
                break

        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()

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
