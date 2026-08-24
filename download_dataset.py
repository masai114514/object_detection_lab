from roboflow import Roboflow

rf = Roboflow(api_key="Eup3H7FRMTneW3JxCbTy")
project = rf.workspace("my-workspace-7j2fi").project("cup-detection-w8kfb")
dataset = project.version(3).download("yolov8")
