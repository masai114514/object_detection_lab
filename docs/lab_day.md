# 机房当天速查卡（Jetson Orin NX + USB 摄像头 + 校园网）

配套文档：docs/jetson_deploy.md（完整版）。本页是"带上板子去机房"当天的一页速查。

## 出门前检查

- [ ] microSD 卡**已刷好 JetPack**（在家用 Etcher 刷，镜像 6-8GB 提前下）
- [ ] `jetson_deploy.tar.gz` 已拷进 U 盘（在仓库根目录，23M）
- [ ] 板子 + 电源适配器 + USB 摄像头
- [ ] HDMI 线（或 DP 线）
- [ ] USB 键盘 + 鼠标
- [ ] 网线（校园网）＋ 插线板（可选）

## 到机房接线

```
显示器(HDMI) ←板子 HDMI口    USB键盘/鼠标 →板子 USB口
网线 → 板子网口              摄像头 → 板子 USB口（识别为 /dev/video0）
microSD 卡 → 板子卡槽        最后插电源
```

显示器若被机房电脑占着：把 Jetson 插进显示器**空闲的 HDMI 口**，按显示器
"信号源/Input"键切到 HDMI。**不需要拔机房电脑的线。**

## 首次开机（约 5-10 分钟）

1. 走 Ubuntu 向导：设用户名/密码、时区选上海
2. 联网：若校园网需**浏览器认证**（登录页），用板子自带的 Firefox 打开任意网页
   登录一次即可；若需 802.1x 再单独配 NetworkManager
3. 终端确认环境：
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ls /opt/ros/        # 有 foxy 或 humble
   ls /dev/video*      # 摄像头在
   ```

## 部署四步

```bash
mkdir -p ~/jetson_app && tar -xzf ~/jetson_deploy.tar.gz -C ~/jetson_app
cd ~/jetson_app/jetson_deploy

# 1) 装 ultralytics（版本钉在与训练一致）
pip3 install -U ultralytics==8.4.127      # 慢就加 -i 清华镜像

# 2) 导出 TensorRT engine（5-10 分钟，FP16）
python3 scripts/export_jetson.py --model models/combined_best.pt --half

# 3) 实时识别（验收：FPS≥5，显示类别/框/置信度）
python3 jetson/detect.py --model models/combined_best.engine
#    有显示器：画面按 s 记录真值、q 退出
#    只插电无屏/SSH：加 --no-show --save-interval 3（终端打印 FPS）

# 4) 离线复跑 20 张测试集（验收：≥80%）
python3 scripts/evaluate.py --model models/combined_best.pt --test-root test_images --conf 0.25
```

## ROS2 发布（验收）

```bash
source /opt/ros/<foxy|humble>/setup.bash
cd ~/jetson_app/jetson_deploy/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run detection_pkg detection_node --ros-args \
    -p model_path:=$HOME/jetson_app/jetson_deploy/models/combined_best.engine \
    -p camera_index:=0
# 另一个终端：
source /opt/ros/<foxy|humble>/setup.bash
ros2 topic echo /detections
```

## 当天高频问题

| 现象 | 处理 |
|------|------|
| 显示器无信号 | 检查信号源是否切到 HDMI；HDMI 线是否插板子的视频口；重插电源重启 |
| 网不通 | 浏览器登录校园网认证页；或用手机热点 + USB WiFi |
| FPS 只有几帧 | 确认加载 `.engine`（不是 `.pt`）且导出时用了 `--half` |
| pip 很慢/失败 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；或让 pip 走镜像 |
| 摄像头打不开 | `ls /dev/video*`；换 USB 口重插 |
| 导出卡住下载 | 推理不触发 AMP/GitHub 下载；卡住多半是网络，加镜像重试 |

## 离开机房前要带走的

- `~/jetson_app/jetson_deploy/results/`（test_results.csv、error_cases/、auto_*.jpg）
- 如果没当场验证完，用 `scp -r results jetson@<IP>:~/` 拷回家再说

> 备注：如果回家还想用 SSH 接着调 Jetson，记下板子 IP；或下次再带显示器去机房。
> 部署包重打：`bash scripts/sync_to_jetson.sh --local`
