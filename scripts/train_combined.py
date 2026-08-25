from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data_combined.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.001,
    patience=10,
    name='combined_model'
)
