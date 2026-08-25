# AutoDL 云端训练指南（RTX 5090）

本地 Mac MPS 训练 yolov8s 约 5 min/epoch，150 epoch 要 8+ 小时；AutoDL 的 RTX 5090
实测 **5.2 it/s（~27 s/epoch）**，150 epoch 约 1 小时，快 ~10 倍。

## 打包与上传

```bash
# 打包（数据集 + 脚本 + yolov8s.pt + 个人测试集）
bash scripts/sync_cloud.sh            # 生成 cloud_train.tar.gz

# 上传（端口/地址在 AutoDL 控制台「SSH 登录」里看）
scp -P <端口> cloud_train.tar.gz root@<地址>:/root/autodl-tmp/
```

## 实例上运行

```bash
cd ~/autodl-tmp && tar -xzf cloud_train.tar.gz
cd object_detection_lab
bash cloud/run_training.sh            # 检查环境 → 训练 → 自动在个人测试集上评估
```

`run_training.sh` 会自动寻找带 torch 的 python（兼容 AutoDL 的 conda 布局）。
实例须选「PyTorch」镜像（自带 torch+CUDA），本实验用的 torch 2.12.1+cu130、ultralytics 8.4.127。

训练完把权重和结果取回本地：

```bash
scp -P <端口> root@<地址>:~/autodl-tmp/object_detection_lab/runs/detect/combined_model_v3/weights/best.pt .
scp -P <端口> root@<地址>:~/autodl-tmp/object_detection_lab/results/test_results.csv .
```

## 遇到的坑（复现时注意）

1. **GitHub 下载被墙**：ultralytics 默认从 github.com 下载权重（yolov8s.pt）和画图字体
   （Arial.ttf），国内网络会卡死。解决办法：
   - `train_combined.py` 的 `ensure_weights()`：本地没有 yolov8s.pt 时从 hf-mirror.com 镜像下载；
   - 字体：从本机 `/System/Library/Fonts/Supplemental/Arial.ttf` scp 到实例
     `/root/.config/Ultralytics/Arial.ttf`，文件存在即跳过下载；
2. **AMP 检查会下载 yolo26n.pt**（仅 CUDA 上）：训练前的 Automatic Mixed Precision 检查会下载
   一个小的 yolo26n.pt 验证 AMP，同样走 GitHub。预先从
   `https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`
   下载到实例的项目目录即可跳过。
3. **AutoDL 是非交互 shell**：`python3` 可能不在 PATH，用 `/root/miniconda3/bin/python3`；
   远程后台任务务必 `setsid nohup ... > log 2>&1 < /dev/null &` 完全脱离终端，否则 SSH 断开会被杀。
4. **pkill 自杀陷阱**：`pkill -f run_training.sh` 会匹配到包含该字符串的远程 shell 自身，
   把执行命令的 shell 杀掉。用自排除写法 `pkill -f "run_training[.]sh"`，或把「杀」和「重启」
   拆成两条 SSH 命令。

## 数据

云包内是 `public_data/balanced`（1500 cup : 1110 mouse，train/val，无交叉重复）。
`data_balanced.yaml` 用相对路径，解压后从项目根目录运行即可。
