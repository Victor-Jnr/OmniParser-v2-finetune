# OmniParser模型训练指南

本指南将教您如何使用现有的OmniParser来生成训练数据，然后训练自己的YOLO图标检测模型和Florence2图标描述模型。

## 训练流程概述

OmniParser使用两个独立但协作的模型：

1. **YOLO模型** (`weights/icon_detect/model.pt`) - 检测图标的边界框
2. **Florence2模型** (`weights/icon_caption_florence`) - 为检测到的图标生成语义描述

## 数据格式说明

### YOLO训练数据格式

```
训练目录/
├── yolo_format/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image1.txt  # 格式: class_id center_x center_y width height
│   │   └── ...
│   ├── val/
│   │   ├── image2.jpg
│   │   ├── image2.txt
│   │   └── ...
│   └── dataset.yaml
```

**YOLO标注文件示例** (image1.txt):

```
0 0.5 0.3 0.1 0.1
0 0.7 0.8 0.05 0.05
```

- 每行一个目标：类别ID 中心x 中心y 宽度 高度（相对坐标0-1）
- 类别0：图标（单类检测）

### Florence2训练数据格式

```json
[
  {
    "bbox": [0.45, 0.25, 0.55, 0.35],
    "content": "Settings button",
    "image_path": "path/to/image.jpg"
  }
]
```

## 步骤1：准备原始图像

将您要用于训练的图像放在一个目录中：

```bash
mkdir raw_images
# 将您的APK截图或其他界面图像复制到此目录
cp /path/to/your/screenshots/* raw_images/
```

## 步骤2：收集训练数据

### 2.1 自动生成训练数据

```bash
python collect_training_data.py \
    --input_dir ./raw_images \
    --output_dir ./training_data \
    --device cuda \
    --box_threshold 0.05
```

### 2.2 带手动校正的训练数据收集

```bash
python collect_training_data.py \
    --input_dir ./raw_images \
    --output_dir ./training_data \
    --device cuda \
    --box_threshold 0.05 \
    --manual_correction
```

这将生成一个`manual_correction.csv`文件，您可以在Excel或其他表格编辑器中打开并修正：


| image_name      | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 | original_content | corrected_content | keep_element |
| --------------- | ------- | ------- | ------- | ------- | ---------------- | ----------------- | ------------ |
| screenshot1.png | 0.1     | 0.1     | 0.2     | 0.2     | "Settings"       | "Settings button" | TRUE         |
| screenshot1.png | 0.3     | 0.3     | 0.4     | 0.4     | "unclear text"   | "Search icon"     | TRUE         |

### 2.3 应用手动校正

```bash
python collect_training_data.py \
    --output_dir ./training_data \
    --apply_corrections
```

### 2.4 若 json 处理失败使用以下命令修复后重新执行 2.3

```
python repair_json_files.py --output_dir ./training_data
```

## 步骤3：训练模型

### 3.1 只训练YOLO模型

```bash
python finetune_omniparser_models.py \
    --mode yolo \
    --data_dir ./training_data \
    --yolo_epochs 100
```

### 3.2 只训练Florence2模型

```bash
python finetune_omniparser_models.py \
    --mode florence2 \
    --data_dir ./training_data \
    --florence_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-5
```

### 3.3 同时训练两个模型

```bash
python finetune_omniparser_models.py \
    --mode both \
    --data_dir ./training_data \
    --yolo_epochs 100 \
    --florence_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-5
```

## 步骤4：使用训练好的模型

训练完成后，模型将保存到：

- YOLO模型：`weights/icon_detect/model.pt`
- Florence2模型：`weights/icon_caption_florence_finetuned/`

### 测试训练好的模型

```python
from demo import main

# 使用训练好的模型处理新图像
main('path/to/test_image.png')
```

## 针对APK界面的特殊优化

### 1. 数据收集策略

```bash
# 针对移动端界面，使用较低的检测阈值
python collect_training_data.py \
    --input_dir ./apk_screenshots \
    --output_dir ./apk_training_data \
    --box_threshold 0.03 \
    --manual_correction
```

### 2. 训练参数调整

```bash
# APK界面通常元素较小，需要更多训练轮次
python finetune_omniparser_models.py \
    --mode both \
    --data_dir ./apk_training_data \
    --yolo_epochs 200 \
    --florence_epochs 10 \
    --batch_size 4 \
    --learning_rate 5e-6
```

### 3. 数据增强建议

在`raw_images`目录中包含：

- 不同分辨率的APK截图
- 不同Android版本的界面
- 不同应用的相似界面元素
- 不同语言的界面（如果需要多语言支持）

## 完整示例工作流程

```bash
# 1. 准备数据
mkdir apk_training_project
cd apk_training_project
mkdir raw_images

# 2. 复制APK截图到raw_images/

# 3. 生成训练数据并进行手动校正
python ../collect_training_data.py \
    --input_dir ./raw_images \
    --output_dir ./training_data \
    --manual_correction

# 4. 在Excel中编辑 training_data/manual_correction.csv

# 5. 应用校正并生成最终训练数据
python ../collect_training_data.py \
    --output_dir ./training_data \
    --apply_corrections

# 6. 训练模型
python ../finetune_omniparser_models.py \
    --mode both \
    --data_dir ./training_data \
    --yolo_epochs 150 \
    --florence_epochs 8

# 7. 测试训练好的模型
python ../demo.py
```

## 训练监控和调优

### YOLO训练监控

- 训练过程中会在`yolo_training/icon_detect/`目录生成日志和图表
- 关注以下指标：
  - `precision`：精确率
  - `recall`：召回率
  - `mAP50`：IoU=0.5时的平均精度

### Florence2训练监控

- 观察训练和验证损失的变化
- 如果验证损失不再下降，可以提前停止训练

### 常见问题解决

1. **YOLO检测精度低**：

   - 降低`box_threshold`（如0.03）
   - 增加训练轮次
   - 检查标注质量
2. **Florence2描述不准确**：

   - 增加训练数据量
   - 改善标注质量
   - 调整学习率
3. **训练速度慢**：

   - 减少batch_size
   - 使用更强的GPU
   - 减少图像分辨率

## 评估模型性能

创建一个测试脚本来评估训练效果：

```python
import os
from demo import main
from pathlib import Path

test_images_dir = "test_images"
for img_path in Path(test_images_dir).glob("*.png"):
    print(f"Processing {img_path.name}...")
    result = main(str(img_path))
    # 分析result并计算准确率
```

通过以上流程，您可以训练出适合您特定应用场景的OmniParser模型。记住，模型的质量很大程度上取决于训练数据的质量和数量。
