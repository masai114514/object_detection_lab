import cv2
import os
from datetime import datetime

class DataCollector:
    def __init__(self, object_class, save_dir='test_images'):
        self.object_class = object_class
        self.save_dir = f'{save_dir}/{object_class}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        self.count = 0

    def collect(self):
        print(f"采集 {self.object_class} 类数据，按 's' 保存，'q' 退出")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            cv2.imshow('Collect', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f'{self.save_dir}/{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                self.count += 1
                print(f"已保存: {filename}")
            elif key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"共采集 {self.count} 张")

if __name__ == '__main__':
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else 'cup'
    DataCollector(cls).collect()
