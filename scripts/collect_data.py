import cv2
import os
from datetime import datetime

class DataCollector:
    def __init__(self, object_class, save_dir='personal_data'):
        self.object_class = object_class
        self.save_dir = f'{save_dir}/{object_class}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        self.count = 0

    def collect(self):
        print(f"采集 {self.object_class} 类数据")
        print("按 's' 保存，'a' 自动连拍（每30帧），'q' 退出")
        auto_mode = False
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            info = f"Class: {self.object_class} | Count: {self.count} | Auto: {auto_mode}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self._save_frame(frame)
            elif key == ord('a'):
                auto_mode = not auto_mode
                print(f"自动模式: {'开' if auto_mode else '关'}")
            elif key == ord('q'):
                break
            if auto_mode:
                frame_count += 1
                if frame_count % 30 == 0:
                    self._save_frame(frame)
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"采集完成，共 {self.count} 张")

    def _save_frame(self, frame):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'{self.save_dir}/{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        self.count += 1
        print(f"已保存: {filename}")

if __name__ == '__main__':
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else 'cup'
    DataCollector(cls).collect()
