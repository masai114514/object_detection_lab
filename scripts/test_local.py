import cv2
import csv
import os
from ultralytics import YOLO

# 模型路径
model = YOLO('runs/detect/combined_model/weights/best.pt')

# 测试图片根目录
test_root = 'test_images'
results = []
class_names = ['cup', 'mouse']  # 顺序与训练时一致

# 遍历所有子目录中的图片
for class_name in class_names:
    class_dir = os.path.join(test_root, class_name)
    if not os.path.exists(class_dir):
        print(f"警告：{class_dir} 不存在")
        continue
    for img_file in sorted(os.listdir(class_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(class_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 推理
        res = model(frame, imgsz=640, verbose=False)[0]
        pred_class = 'none'
        conf = 0.0
        if res.boxes is not None and len(res.boxes) > 0:
            # 取置信度最高的框
            best_conf_idx = res.boxes.conf.argmax()
            pred_class = res.names[int(res.boxes.cls[best_conf_idx])]
            conf = float(res.boxes.conf[best_conf_idx])

        true_class = class_name
        correct = (pred_class == true_class)
        results.append({
            'image': img_file,
            'true_class': true_class,
            'pred_class': pred_class,
            'confidence': round(conf, 3),
            'correct': correct
        })
        print(f"{img_file}: true={true_class}, pred={pred_class}, conf={conf:.3f}, correct={correct}")

# 保存 CSV
output_csv = 'test_results_local.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image', 'true_class', 'pred_class', 'confidence', 'correct'])
    writer.writeheader()
    writer.writerows(results)

# 统计
total = len(results)
correct_count = sum(1 for r in results if r['correct'])
accuracy = correct_count / total if total > 0 else 0
print(f"\n总测试图片数: {total}")
print(f"正确数: {correct_count}")
print(f"识别率: {accuracy:.1%}")
