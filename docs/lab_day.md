# 机房当天速查卡（Jetson Orin NX + USB 摄像头 + 校园网）

配套文档：docs/jetson_deploy.md（完整版）。本页是"带上板子去机房"当天的一页速查。

## 出门前检查

- [ ] **板子系统由实验室预装**（老师确认过，无需自己刷卡）
      —— 但请向老师确认：① 启动账户名/密码；② 是否已含 JetPack 的
      torch / OpenCV / ROS2（见下「如果系统已预装」）
- [ ] `jetson_deploy.tar.gz` 已拷进 U 盘（在仓库根目录，23M）
- [ ] 板子 + 电源适配器 + USB 摄像头
- [ ] HDMI 线（或 DP 线）
- [ ] USB 键盘 + 鼠标
- [ ] 网线（校园网）＋ 插线板（可选）

> 如果板子其实没系统（或想留备份），才需要 microSD 64GB+ + Etcher 刷 JetPack
> （方法见 docs/jetson_deploy.md 第 1 节）。有系统则跳过，省一大步。

## 如果系统已预装

```bash
# 开机后用老师给的账户登录，2 秒确认环境：
cat /etc/nv_tegra_release           # 有输出 = JetPack 正常
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls /opt/ros/                        # 有 foxy 或 humble = ROS2 就位
ls /dev/video*                      # 摄像头在
```
- 若 torch/ROS2 都有 → 直接跳「部署四步」，只装 ultralytics 即可；
- 若只是裸 Ubuntu 没有 torch → 装 JetPack 或 NVIDIA 提供的 torch（较麻烦，
  先找老师确认系统镜像是不是 JetPack）；

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

# 2) 推理引擎：直接 torch FP16（学校镜像缺 tensorrt，engine 导不了，见下）
#    python3 scripts/export_jetson.py --model models/combined_best.pt --half  # 仅完整JetPack可用

# 3) 实时识别（验收：FPS≥5，显示类别/框/置信度；实测 torch FP16 约 24-25 FPS）
python3 jetson/detect.py --model models/combined_best.pt --half
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
    -p model_path:=$HOME/jetson_app/jetson_deploy/models/combined_best.pt \
    -p camera_index:=0 -p device:=0 -p half:=true -p show:=false
# 另一个终端：
source /opt/ros/<foxy|humble>/setup.bash
ros2 topic echo /detections
```

## 当天高频问题

| 现象 | 处理 |
|------|------|
| 显示器无信号 | 检查信号源是否切到 HDMI；HDMI 线是否插板子的视频口；重插电源重启 |
| 网不通 | 浏览器登录校园网认证页；或用手机热点 + USB WiFi |
| FPS 只有几帧 | 确认加了 `--half` 且用 GPU（`--device 0`）；`.pt` 别省 `--half` |
| pip 很慢/失败 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；或让 pip 走镜像 |
| 摄像头打不开 | `ls /dev/video*`；换 USB 口重插 |
| `import tensorrt` 失败 | 学校镜像缺 `libnvdla_compiler.so`，**不要尝试装 tensorrt**；直接用 `.pt --half` 跑（24-25 FPS） |

## 离开机房前要带走的

- `~/jetson_app/jetson_deploy/results/`（test_results.csv、error_cases/、auto_*.jpg）
- 如果没当场验证完，用 `scp -r results jetson@<IP>:~/` 拷回家再说

> 备注：如果回家还想用 SSH 接着调 Jetson，记下板子 IP；或下次再带显示器去机房。
> 部署包重打：`bash scripts/sync_to_jetson.sh --local`
