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

### 2.3.1 若 json 处理失败使用以下命令修复后重新执行 2.3

```
python repair_json_files.py --output_dir ./training_data
```

### 2.4 删除一部分老数据 old_percentage 20 为保存 20% 的老数据, 其它为新数据, 适当增加保存量
```
python collect_training_data.py --output_dir ./training_data --apply_corrections --old_percentage 20
```


### 2.5 数据增强功能（提升模型性能）

项目现在包含先进的数据增强系统，通过图像变换技术扩展训练数据集，显著提升模型泛化能力和性能。

#### 数据增强技术

系统采用智能随机策略：首先随机选择1-2种增强方法，然后在每种方法内随机确定具体参数，应用以下图像变换：

- **旋转变换**：±15度范围内随机旋转（保持文本可读性）
- **裁剪变换**：0.5-3%边距的随机裁剪（保持UI元素完整性）
- **亮度调整**：±20像素值的亮度变化（模拟不同光照条件）
- **对比度调整**：0.8x-1.2x对比度缩放（模拟不同显示设备）
- **高斯噪声**：添加轻微高斯噪声（模拟图像质量变化）
- **随机缩放**：-10%到+10%的图像缩放（模拟不同分辨率设备）

**注意**：已移除图片翻转功能，因为UI元素的翻转不能有效提升识别能力，反而可能误导模型学习。

**随机逻辑**：先随机选择增强对象（1-2种方法），再在选中的方法内随机确定变换幅度，确保训练数据的多样性和真实性。

#### 使用方法


**独立运行数据增强**

```bash
# 对训练数据进行3倍增强: 原始数据 *3 的数据量, 随机在以上选项中处理
python data_augmentation.py --data_dir training_data/florence_format --multiplier 3

# 自定义目录和5倍增强
python data_augmentation.py --data_dir /path/to/your/data --multiplier 5
```

#### **数据清理**

增强后，您可能需要手动查看并删除 `imgs/` 目录中不满意的图片。使用清理选项自动删除对应的JSON条目：

```bash
# 手动删除不需要的图片后，清理JSON数据
python data_augmentation.py  --data_dir training_data/florence_format --clean_missing

# 指定目录的数据清理
python data_augmentation.py --data_dir /path/to/your/data --clean_missing
```

**推荐的数据调整工作流程**：
1. 运行数据增强：`python data_augmentation.py --multiplier 3`
2. 检查生成的图片：`training_data/florence_format/imgs/`
3. 手动删除不满意的增强图片（保留质量好的）
4. 运行数据清理：`python data_augmentation.py --clean_missing`
5. 使用清理后的数据集进行训练

#### 推荐设置

| 数据集规模 | 推荐增强倍数 | 预期训练时间增加 | 性能提升期望 |
|-----------|-------------|-----------------|-------------|
| < 50 样本 | 5-10倍 | +200-400% | 显著提升 |
| 50-200 样本 | 3-5倍 | +100-200% | 明显提升 |
| > 200 样本 | 2-3倍 | +50-100% | 适度提升 |

#### 输出结构

增强后的数据结构：

```
training_data/florence_format/
├── florence_data.json              # 更新的训练数据（含增强样本）
├── florence_data_original.json     # 原始数据备份
└── imgs/                           # 增强图像文件夹
    ├── example_aug_1_rot_5.2.png
    ├── example_aug_1_bright_-10_noise_10.png
    └── ...
```

#### 技术细节

- **处理逻辑**：先裁剪bbox区域，再对裁剪后的图像进行增强变换
- **坐标更新**：增强后的图像bbox设置为[0,0,1,1]（全图坐标）
- **图像尺寸标准化**：所有裁剪图像调整为64x64像素（遵循OmniParser标准）
- **坐标准确性**：无论如何变换，坐标始终指向正确的UI元素内容
- **自动备份**：原始训练数据自动备份，确保数据安全
- **错误处理**：robust错误处理机制，处理损坏或缺失图像

#### 质量控制

- 裁剪限制防止内容丢失（最大3%边距）
- 旋转限制保持文本可读性（±15°）
- 亮度/对比度范围维持可见性
- 文件名格式显示应用的变换：`original_aug_N_technique_params.png`

#### 性能优势

- **改善泛化能力**：模型从多样化视觉条件中学习
- **减少过拟合**：更多样化的训练样本
- **提高鲁棒性**：处理光照、旋转和噪声变化
- **提升准确率**：特别有效于小数据集场景

详细使用说明请参考：`DATA_AUGMENTATION_README.md`

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
    --florence_epochs 20 \
    --batch_size 4 \
    --learning_rate 5e-5
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
