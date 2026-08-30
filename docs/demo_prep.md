# 现场演示讲解准备 —— 实验一：目标检测与识别

> 助教可能随时指着代码问「这行在干嘛」，也会抽查关键数字。下面按「开场白 → 端到端流程 →
> 逐文件要点 → 高频问答 → 数字记忆卡 → 现场脚本」组织。标注 ⭐ 的是最可能被问到的地方。

---

## 1. 一分钟开场白（建议背熟）

> 本实验用 **YOLOv8** 做桌面上**杯子（class 0）和鼠标（class 1）**的双类目标检测。
> 数据 = **Roboflow 公开数据（cup v3 + Computer-Mouse v2，CC BY 4.0）** + **个人采集标注的 20 张测试图**，
> 合并成 2610 张平衡训练集（1500 cup : 1110 mouse）训练 yolov8s，验证集 **mAP50 = 0.975**。
> 在 **Jetson Orin NX** 上用 torch FP16 CUDA 实时推理，实测 **24~25 FPS**（要求 ≥5 FPS），
> 画面实时显示类别、检测框、置信度，并通过 **ROS2 把结果发布到 /detections 话题**。
> 个人测试集 **20 个物体正确识别 19/20 = 95%**（cup 10/10、mouse 9/10），满足 ≥80%。

---

## 2. 端到端流程（先讲这条主线，再展开）

```
公开数据(Roboflow) ─┐
                    ├─> build_combined_dataset.py 合并平衡 ─> data_balanced.yaml
个人采集(20张测试) ─┘
                    └─> train_combined.py (yolov8s 迁移学习) ─> best.pt (mAP50=0.975)
                         │
                         ├─> evaluate.py  离线评估测试集  ─> test_results.csv + error_cases/
                         ├─> detect.py    Jetson 实时识别 ─> 画面 + FPS + 录制 demo_detect.mp4
                         └─> detection_node.py ROS2 发布 ─> /detections (JSON)
```

四件事：**备数据 → 训练 → 真机推理 → ROS2 发布**。每一件事对应一个脚本，讲代码时按这条线走。

---

## 3. 逐文件讲解要点（TA 指着哪行讲哪段）

### scripts/download_dataset.py —— 数据下载 ⭐
- 从 Roboflow 下载 cup v3 / mouse v2（v15 可选），输出到 `public_data/`。
- 设计点：**API key 从环境变量 `ROBOFLOW_API_KEY` 读取，不硬编码**——避免提交到 GitHub 泄露凭据。
- 若被问「为什么不用现成的」：Roboflow 数据集免费、已带 YOLO 标注，CC BY 4.0 可商用。

### scripts/build_combined_dataset.py —— 构建平衡数据集 ⭐
- `collect_samples()`：递归扫描图片目录，**标签目录由 `_sibling_labels_dir` 推导**（兼容
  扁平 / Roboflow / 拆分三种布局），并把类别号统一重映射（mouse 源 → class 1）。
- **类别不平衡**：cup 有 3191 张，鼠标只有 1110，于是 `--max-cups 1500` 把多数类随机截断。
- **分层划分**：每类独立做 85/15 切分，保证 train/val 里 cup 和 mouse 都有。
- ⭐ 值得强调的 bug 修复：写输出前 `shutil.rmtree` 清空旧目录，**避免上一次运行残留导致
  train/val 交叉重复（数据泄漏）**；加 `cu_/mo_` 前缀避免杯鼠重名。
- 输出 `data_balanced.yaml`（相对路径，Mac / Jetson 通用）。

### scripts/train_combined.py —— 训练 ⭐
- **迁移学习**：`yolov8s.pt` 预训练权重（在 COCO 上学过通用特征），`lr0=0.01` 迁移标准值。
- `ensure_weights()`：本地无权重时从 **hf-mirror.com 镜像下载**——GitHub Releases 在国内常被
  屏蔽导致 ultralytics 自动下载卡死（「已知问题」里的一条实测坑）。
- `pick_device()`：CUDA → MPS → CPU 自动选，所以 Mac 也能训。
- 超参数：**150 epoch、cos_lr 余弦退火、patience=30 早停、imgsz=640、batch=16**。
- 训练环境：AutoDL RTX 5090 云 GPU ~70 min；本地 Mac MPS 约 8 h。

### scripts/evaluate.py —— 离线评估测试集 ⭐
- 每张测试图取**置信度最高的框**作判定；判定正确 = 检测到物体 **且** 类别正确 **且** conf ≥ 0.25。
- 输出按类准确率、混淆情况（true → pred，含 'none'）、`test_results.csv`、错误案例复制到
  `results/error_cases/`。
- 实测：**cup 10/10、mouse 9/10、总体 19/20 = 95%**。唯一错误案例 `a79a99a9...`：竖拍、目标
  贴顶出框，模型 conf 0.499 误判为 cup。

### jetson/detect.py —— Jetson 实时识别 ⭐
- 主循环五步：**读帧 → YOLO 推理 → 画框+类别+置信度 → 叠加 FPS → 可选录制**。
- FPS 用**滑动平均**：`fps = 0.9*fps + 0.1*(1/Δt)`，显示更稳。
- `--half`：FP16 推理，torch CUDA 上从 ~10 FPS 提到 **24~25 FPS**（够 ≥5 要求）。
- `--record --duration 70`：到点正常收尾，`vwriter.release()` 把 mp4 索引写全。
  ⭐ **不要用 `timeout` 强杀**——SIGTERM 直接杀死进程、moov 索引缺失，mp4 打不开（实测坑）。
- 按 `s` 保存当前帧 + 输入真值(0/1)，记录到 CSV；预测≠真值时自动存 `error_cases/`。
- `--no-show`：SSH 无显示时 headless 模式，自动周期存帧 + 打印 FPS。

### ros2_ws/src/detection_pkg/detection_node.py —— ROS2 发布节点 ⭐
- 继承 `rclpy.Node`，`declare_parameter` 声明 8 个可调参数（model_path / camera_index / conf /
  imgsz / device / half / topic / show）。
- 主循环把每帧的 `[{'class','confidence','bbox'}, ...]` 用 `json.dumps` 序列化，
  `std_msgs/String` 发布到 **/detections**（QoS depth 10）。
- 复用 detect.py 的画框/存帧/错误案例逻辑；无 DISPLAY 自动切 headless。
- 构建与验证：
  ```bash
  cd ros2_ws && colcon build --symlink-install && source install/setup.bash
  ros2 run detection_pkg detection_node --ros-args -p model_path:=models/combined_best.pt -p half:=true
  # 另一终端：ros2 topic echo /detections
  ```

### scripts/export_jetson.py —— TensorRT 导出（可选加分点）
- `.pt` → `.engine`（FP16），Orin NX 上可达 **30+ FPS**。
- ⭐ 本实验**没用** engine：学校镜像缺 `libnvdla_compiler.so`，`import tensorrt` 直接失败，
  apt / 清华源 / NGC 都拿不到 → 退而求其次用 **torch FP16**（24~25 FPS），已远超 5 FPS 要求。
  如果助教问，答：「engine 是优化项，不是必需项；当前方案满足全部验收指标」。

---

## 4. 高频 Q&A（先自己复述一遍）

| 问题 | 回答要点 |
|------|----------|
| 为什么用 YOLOv8？ | 一阶段检测器快、预训练迁移学习上手快、ultralytics 生态成熟（导出/ROS2/评测齐全），适合小数据双类任务 |
| 类别不平衡怎么办？ | cup 截断到 1500、mouse 全量 1110；再分层划分，保证两类在 train/val 都有代表 |
| 数据从哪来？版权？ | Roboflow cup-detection v3 + Computer-Mouse v2，均 CC BY 4.0；个人采集 20 张并标注 |
| 个人数据怎么标注的？ | 模型辅助：best.pt 高置信框自动标注 19 张 + 人工校正 1 张（贴顶出框那例）；复训前建议 labelImg 抽查 |
| 为什么测试集 mouse 只有 90%？ | 公开数据与个人桌面存在域差异；错误例是竖拍+目标部分出框+低置信度 0.5 |
| 为什么不用 TensorRT？ | 镜像缺 DLA 编译器库（libnvdla_compiler.so），三种途径都拿不到；torch FP16 已 24~25 FPS |
| 训练在哪跑的？ | AutoDL RTX 5090 云 GPU ~70min；Mac MPS 本地也能训（~8h） |
| 视频为什么还要重封装？ | 录制写入约 12 FPS 但容器标 20 FPS → 播放 1.63× 快进；`scripts/remux.py` 重封装为 12.24fps 还原真实速度 |
| 判定规则？ | 每张图一个已知物体，取最高置信框；类别对 + conf≥0.25 才算正确 |
| ROS2 节点和 detect.py 什么关系？ | detect.py 独立程序；detection_node 是 ROS2 版，把结果发布成话题，供 `ros2 topic echo` 等任意节点订阅 |
| GitHub 为什么分这么多次提交？ | 老师要求增量提交、禁止单次全量；每个功能/修复独立提交，可追溯 |

---

## 5. 关键数字记忆卡（考前过一遍）

- 类别：`nc=2`，**0=cup / 1=mouse**
- 训练集 **2610 张 = 1500 cup : 1110 mouse**；train **2219** / val **391**（85:15）
- 个人测试集 **20 张 = 10 cup + 10 mouse**，正确率 **19/20 = 95%**
- 模型 **yolov8s**，150 epoch，imgsz 640，**mAP50 = 0.975**
- Jetson 实测 **24~25 FPS**（torch FP16 CUDA，要求 ≥5）；engine 理论 30+
- 演示视频 **70 s / 857 帧 / 12.24 fps**；逐帧校验：杯框 60 s、鼠框 60 s、同框 51 s、FPS 全程叠加
- 模型权重 22.5 MB；ROSO2 话题 **/detections**（std_msgs/String，JSON）
- 唯一错误案例：`mouse_x_cup_a79a99a9...jpg`（鼠标竖拍贴顶出框，conf 0.5 → 误判 cup）
- GitHub **52+ 次增量提交**（无单次全量），本地与远程一致

---

## 6. 现场演示脚本（5 分钟版）

1. **开场**（30 s）：说开场白，播一遍 `04_结果视频` 里的演示视频。
2. **数据**（1 min）：打开 `test_images/{cup,mouse}` 随便挑几张，讲标注 txt 格式
   `class cx cy w h`（归一化坐标）；提一句 2610 张平衡训练集的构成。
3. **训练与评估**（1 min）：展示 `train_combined.py` 命令 + `results.csv` 里 mAP50，
   以及 `evaluate.py` 输出的 cup 100% / mouse 90% / 总体 95%。
4. **Jetson 实时**（1 min）：跑 `python3 jetson/detect.py --model ... --half`，
   现场把杯子、鼠标放进画面，指出同框、置信度、FPS 实时变化。
5. **ROS2**（1 min）：另一终端 `ros2 topic echo /detections`，切到发布终端动一动鼠标，
   让订阅端实时跳出 JSON，说明「发布-订阅解耦」。
6. **结果与错误案例**（30 s）：打开 `test_results.csv` 和 `error_cases/` 里的唯一错误图，
   解释「竖拍 + 贴顶出框 + 低置信」的失败模式。
7. **收尾**（30 s）：GitHub 仓库 + 增量提交历史；问一句「要不要看下载/构建脚本」主动展示。

**现场救场**：忘词就看「5. 数字记忆卡」；被问住就说「这个我项目里遇到过，是……」（引导到
「7. 实测坑与改进」里的真实 bug，反而加分）。

---

## 7. 实测坑与改进（讲出来是加分项）

1. **旧数据集缺鼠标标注**：3191 张杯子 / 0 张鼠标 → 旧模型把鼠标全识别成杯子（10 只 0 命中）。
   改用 Roboflow Computer-Mouse 重建平衡数据集修复。
2. **录制 mp4 损坏**：`timeout 75` 强杀使 `moov` 索引缺失、文件打不开 → 加 `--duration` +
   SIGTERM 优雅收尾，正常 `release()` 写全索引。
3. **播放 1.63× 快进**：录制 ~12 FPS 写入、容器标 20 FPS → `scripts/remux.py` 按 12.24fps 重封装。
4. **分析脚本掩码 bug**：鼠标框是 BGR `(255,128,0)`，掩码 R 通道范围写错导致误报 0 命中 →
   修正为 `(195,68,0)-(255,188,60)`。
5. **GitHub Releases 下载卡死**：ultralytics 自动下权重被墙 → `hf-mirror.com` 镜像兜底。
6. **download_dataset.py 泄露 API key**：曾硬编码 Roboflow key 提交 → 改为环境变量读取并移除。
