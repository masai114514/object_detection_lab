# 个人数据采集与标注指南（减少域差异）

> **当前状态**：`personal_data/{cup,mouse}/{images,labels}/` 已放入了 20 张个人
> 测试图的镜像及其 YOLO 标注（模型辅助自动标注 + 1 张人工校正），作为已采集标注的
> 个人数据素材。如需并入训练集复训，按本文第 3 节把它作为一个数据源参与合并即可。

公开数据集（Roboflow/COCO）里的杯子和鼠标，和你桌面上拍照的场景存在**域差异**
（光照、桌面材质、拍摄角度、摄像头分辨率都不同）。仅用公开数据训练的模型，测试时
容易在个人图片上失准。解决方法是采集一批**你自己的**图片补充训练。

## 1. 采集

```bash
# 采集训练用图片（保存到 personal_data/{cup,mouse}/）
python3 scripts/collect_data.py --object cup   --mode train
python3 scripts/collect_data.py --object mouse --mode train

# 采集测试用图片（保存到 test_images/{cup,mouse}/，不参与训练）
python3 scripts/collect_data.py --object cup   --mode test
python3 scripts/collect_data.py --object mouse --mode test
```

按 `s` 保存当前帧，按 `q` 退出。

**每类建议 100+ 张**，并刻意变化以下因素（越多越好，决定模型在你桌上的泛化能力）：

| 因素 | 变化建议 |
|------|----------|
| 角度 | 俯拍、侧拍、平视；左右旋转 |
| 距离 | 近景、中景、远景（物体占画面 20%~80%）|
| 光照 | 台灯、自然光、逆光、昏暗 |
| 背景 | 空桌面、键盘旁、书本上、不同颜色桌垫 |
| 遮挡 | 部分被本子/手遮挡（训练增强鲁棒性）|
| 组合 | 杯子和鼠标同时出现在画面里（双类同框）|

## 2. 标注（YOLO 格式）

用 labelImg（PyPI: `pip install labelImg`）或 [Roboflow Annotate](https://app.roboflow.com/) 标注：

- 类别名：杯子和鼠标统一叫 `cup` / `mouse`（最终 class 0=cup, 1=mouse）
- 导出为 **YOLO txt 格式**，目录结构：
  ```
  personal_data/
  ├── cup/images/xxx.jpg      +  cup/labels/xxx.txt      # 每行: 0 cx cy w h
  └── mouse/images/xxx.jpg    +  mouse/labels/xxx.txt    # 每行: 1 cx cy w h
  ```

> 提示：如果只对公开数据集模型做个**快速验证**，可以先跳过标注，用
> `scripts/evaluate.py` 直接测 `test_images/` 看基线，再决定要不要补数据。

## 3. 合并进训练集

把个人数据作为另一个数据源参与合并：

```bash
# cup 源 / mouse 源都可以是「公开数据目录 + 个人目录」的组合。
# 简单做法：把个人标注目录当做一个源，跑两次再合并，或直接把两个源合到一个目录：
mkdir -p public_data/personal_combined
cp -r personal_data/cup/*    public_data/personal_combined/   # 但注意类别号需为 0
cp -r personal_data/mouse/*  public_data/personal_combined/   # 类别号需为 1
```

然后重跑 `build_combined_dataset.py`，把合并后的目录作为 cup-src / mouse-src，
并调大 `--max-cups`（个人数据越多，越能让模型记住你桌面的风格）。

```bash
python3 scripts/build_combined_dataset.py \
    --cup-src   public_data/personal_combined \   # 含公开+个人杯数据
    --mouse-src public_data/personal_combined \   # 含公开+个人鼠标数据
    --out public_data/balanced \
    --max-cups 2500 --val-split 0.15
```

> `collect_samples` 会按目录递归收集，类别号统一重映射为 cup=0 / mouse=1，
> 所以把杯子和鼠标标成 0/1 即可混在同一个目录里。

## 4. 复训

```bash
python3 scripts/train_combined.py      # 用新的 balanced 数据集重训
python3 scripts/evaluate.py --model runs/detect/combined_model_v3/weights/best.pt
```

对比复训前后 `results/test_results.csv` 里 cup / mouse 各自的准确率，应明显提升。
