# Jetson Orin NX 真机部署手册

本文档是实验一的真机部署步骤，配套包 `jetson_deploy.tar.gz`（由
`bash scripts/sync_to_jetson.sh --local` 生成）。包内容：

```
jetson_deploy/
├── models/combined_best.pt     # 训练好的双类权重（cup + mouse）
├── scripts/export_jetson.py    # 导出 TensorRT engine（在 Jetson 上跑）
├── scripts/evaluate.py         # 离线复跑 20 张测试集，统计正确率
├── jetson/detect.py            # 摄像头实时识别（FPS + 存帧 + 交互）
├── test_images/{cup,mouse}     # 个人测试集 20 张（离线验收用）
└── ros2_ws/src/detection_pkg/  # ROS2 发布节点
```

## 0. 前置条件（一次性的）

| 项 | 要求 |
|----|------|
| JetPack | ≥ 5.1（Orin NX 出厂一般自带 JetPack 5.1+/6.x） |
| Python | 3.8+（JetPack 自带） |
| 摄像头 | USB 摄像头最省事；CSI 相机需要额外 gstreamer 参数（见文末） |
| 显示 | 接显示器，或用 SSH + X11 转发；都不行就用 `--no-show`（自动存帧） |
| ROS2 | 如需 ROS2 节点：JetPack 5.x 装 `ros-foxy`，6.x 装 `ros-humble` |

检查：`cat /etc/nv_tegra_release`（JetPack 版本）、`ls /dev/video*`（摄像头）、
`ls /opt/ros/`（ROS2 是否已装）。

## 1. 把包拷到 Jetson

**方式 A：scp**（Jetson 与本机同一局域网）
```bash
# 在本机执行（jetson 用户与 IP 按实际改，Jetson 默认用户一般是 jetson/nvidia）
scp jetson_deploy.tar.gz jetson@192.168.x.x:~/
```

**方式 B：U 盘** —— 直接把 `jetson_deploy.tar.gz` 拷进 U 盘，插到 Jetson，
挂载后 `cp` 到家目录。

## 2. 解包 + 装依赖（Jetson 上执行）

```bash
mkdir -p ~/jetson_app && tar -xzf ~/jetson_deploy.tar.gz -C ~/jetson_app
cd ~/jetson_app/jetson_deploy

# 检查 torch/CUDA（JetPack 一般已带 torch，不要随便覆盖）
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 若没有 torch：JetPack 6 推荐用 NVIDIA 容器；简单起见：
#   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/jp... （见官方 jetpack 说明）

# 装 ultralytics（不会覆盖已装的 torch，放心）
# 版本钉在与训练环境一致（8.4.127），避免版本升级改变 engine 导出行为
pip3 install -U ultralytics==8.4.127
# 国内网络慢可加镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 导出 TensorRT engine（5~10 分钟，只做一次）

```bash
cd ~/jetson_app/jetson_deploy
python3 scripts/export_jetson.py --model models/combined_best.pt --half
# 产出 models/combined_best.engine
```

- `--half` 是 FP16，Orin NX 上 yolov8s@640 实测 30+ FPS，远超 5 FPS 要求。
- 首次导出要装 ONNX（自动），走 PyPI，国内慢就加镜像。
- 若导出报 AMP/yolo26n 下载卡死（GitHub 被墙）：导出的推理流程不走 AMP 检查，
  此坑只在训练时出现；真遇到下载卡住，用 `Ctrl-C` 跳过或用代理预下载。

## 4. 实时识别

**有显示器（推荐演示用）**：
```bash
python3 jetson/detect.py --model models/combined_best.engine
#   按 s：保存当前帧 + 输入真值(0=cup,1=mouse) 记录到 results/test_results.csv
#   按 q：退出
```

**SSH 无显示**：
```bash
python3 jetson/detect.py --model models/combined_best.engine --no-show --save-interval 3
# 每 3 秒存一张带框帧到 results/auto_*.jpg，并在终端打印 FPS 与检测结果
```

FPS 显示在画面左上角（`--no-show` 时打印在终端）。验收要求 ≥5 FPS，
实测 TensorRT FP16 下 30+，**如果明显低于 5 FPS**：确认加载的是 `.engine` 而不是 `.pt`，
且用了 `--half`。

## 5. 离线复跑 20 张测试集（验收：≥80%）

```bash
python3 scripts/evaluate.py --model models/combined_best.pt --test-root test_images --conf 0.25
```
逐类输出 cup/mouse 正确率与混淆情况，结果写 `results/test_results.csv`，
错误案例进 `results/error_cases/`。预期与训练机一致：cup 100% / mouse 90% / 总体 95%。

## 6. ROS2 发布节点（可选，但验收要求 ROS2 发布）

```bash
source /opt/ros/<distro>/setup.bash        # humble 或 foxy，按实际
cd ~/jetson_app/jetson_deploy/ros2_ws
colcon build --symlink-install
source install/setup.bash

# 终端 1：启动检测节点
ros2 run detection_pkg detection_node --ros-args \
    -p model_path:=$HOME/jetson_app/jetson_deploy/models/combined_best.engine \
    -p camera_index:=0

# 终端 2：订阅结果（JSON：类别 / 置信度 / 检测框）
ros2 topic echo /detections
```

参数：`model_path`、`camera_index`、`conf`、`imgsz`、`topic`、`out_dir`、`show`。
无显示器时自动切 headless（只发布 + 打印 FPS，不弹窗）；SSH 无显示时若想彻底不画图，
加 `-p show:=false`。

## 常见问题

1. **`cv2.imshow` 报 "cannot open display"**：Jetson 无显示器且没开 X11 转发。
   加 `--no-show`（ROS2 加 `-p show:=false`）。
2. **摄像头打不开**：先 `ls /dev/video*` 确认设备号；USB 摄像头通常 `0`。
   CSI 相机（如 IMX219）需要 gstreamer 管线，detect.py 支持 `--source` 传管线字符串：
   `--source 'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGR ! appsink'`
3. **ROS2 `colcon build` 找不到命令**：没有 source ROS2 环境，先 `source /opt/ros/<distro>/setup.bash`。
4. **导出 engine 报错 / 很慢**：确认当前 python 里有 `tensorrt`（JetPack 自带；
   没有就 `pip3 install tensorrt`），engine 构建需要几分钟，属正常。
5. **输入真值用 `input()` 卡住**：有显示模式按 `s` 后会在终端等你输入 0/1；
   先切到终端输入再回车，窗口会短暂停住，正常现象。嫌麻烦就用 `--no-show` + 离线 evaluate。
6. **FPS 只有几帧**：确认用 `.engine` + `--half`；如果是 `.pt`，Orin NX 上 CPU 推理很慢。
7. **`/detections` 话题为空**：摄像头没画面就看不到框；先单独跑 detect.py 确认相机正常。
