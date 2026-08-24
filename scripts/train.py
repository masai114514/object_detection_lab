from ultralytics import YOLO
import torch

# 检查 MPS 是否可用（Mac 苹果芯片加速）
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载预训练权重（从 COCO 预训练模型微调）
model = YOLO('yolov8n.pt')

# 开始训练
model.train(
    data='data.yaml',        # 数据集配置文件
    epochs=50,               # 训练轮数，可根据情况调整
    imgsz=640,               # 输入图像尺寸
    batch=8,                 # 批大小，若内存不足可改为 4 或 2
    device=device,           # 使用 MPS 或 CPU
    lr0=0.001,               # 初始学习率
    patience=10,             # 早停耐心值
    name='cup_model'         # 训练名称，结果保存在 runs/detect/cup_model
)
