# K-Fold交叉验证训练说明

## 概述

本项目提供了两个改进的LoRA微调脚本：

1. **finetune_omniparser_lora.py** - 改进的单次训练，使用随机图像分割
2. **finetune_omniparser_lora_kfold.py** - K-Fold交叉验证训练

## 主要改进

### 1. 随机图像分割 (finetune_omniparser_lora.py)

**问题解决:**
- 原始代码使用简单的顺序分割（前80%训练，后20%验证）
- 可能导致同一图像的样本同时出现在训练和验证集中
- 验证集可能不能代表整体数据分布

**改进方案:**
- 基于图像的随机分割，确保同一图像的所有样本要么在训练集，要么在验证集
- 避免数据泄露，提高验证的有效性
- 保持数据分布的代表性

### 2. K-Fold交叉验证训练

**优势:**
- 充分利用所有数据进行训练和验证
- 获得更稳定的模型性能评估
- 减少单次数据分割的随机性影响
- 自动选择最佳模型

## 使用方式

### 单次训练（改进版）

```bash
# 基本训练
python finetune_omniparser_lora.py

# 自定义参数
python finetune_omniparser_lora.py --epochs 20 --batch_size 8 --lr 1e-4

# 训练后合并模型
python finetune_omniparser_lora.py --merge --merge_path weights/my_merged_model
```

### K-Fold交叉验证训练

```bash
# 自动选择最优K值（推荐）
python finetune_omniparser_lora_kfold.py

# 指定K值
python finetune_omniparser_lora_kfold.py --k_folds 5

# 自定义训练参数
python finetune_omniparser_lora_kfold.py --k_folds 5 --epochs 15 --batch_size 16
```

## K值选择指导

基于数据量自动计算：

| 唯一图像数 | 推荐K值 | 说明 |
|-----------|---------|------|
| ≥80 | 5 | 标准5折交叉验证 |
| 50-79 | 4 | 4折交叉验证 |
| 30-49 | 3 | 3折交叉验证 |
| <30 | 2 | 简单2折验证 |

**当前数据集：**
- 总样本数：245
- 唯一图像数：88
- **推荐K值：5**

## 输出文件

### 单次训练输出

```
weights/
├── icon_caption_florence_lora_finetuned/    # LoRA适配器
├── icon_caption_florence_merged/            # 合并后的完整模型（可选）
└── icon_caption_florence_8bit_lora_finetuned/  # 8bit量化模型（可选）
```

### K-Fold训练输出

```
weights/
├── icon_caption_florence_lora_fold_1/       # 每个fold的模型
├── icon_caption_florence_lora_fold_2/
├── icon_caption_florence_lora_fold_3/
├── icon_caption_florence_lora_fold_4/
├── icon_caption_florence_lora_fold_5/
├── icon_caption_florence_lora_kfold_best/   # 最佳模型（自动选择）
└── kfold_results/
    └── kfold_results_5fold.json            # 详细训练结果
```

## 训练结果分析

K-Fold训练完成后会自动生成详细分析：

```
=== K-Fold 训练结果分析 ===
验证损失统计:
  平均值: 2.1234 ± 0.0567
  最小值: 2.0891 (Fold 3)
  最大值: 2.1876 (Fold 1)

各Fold详细结果:
  Fold 1: 2.1876 (196 train, 49 val)
  Fold 2: 2.1345 (195 train, 50 val)
  Fold 3: 2.0891 (194 train, 51 val)
  Fold 4: 2.1123 (195 train, 50 val)
  Fold 5: 2.1234 (196 train, 49 val)

🏆 最佳模型: Fold 3 (验证损失: 2.0891)
模型稳定性 (CV): 0.0267 (稳定)
```

## 推荐工作流

### 1. 快速验证
```bash
# 使用改进的单次训练快速验证
python finetune_omniparser_lora.py --epochs 10
```

### 2. 生产训练
```bash
# 使用K-Fold获得最佳模型
python finetune_omniparser_lora_kfold.py --epochs 15
```

### 3. 模型合并（如需要）
```bash
# 合并最佳K-Fold模型
python finetune_omniparser_lora.py --merge_only \
  --lora_path weights/icon_caption_florence_lora_kfold_best \
  --merge_path weights/icon_caption_florence_kfold_merged
```

## 性能对比

| 方法 | 训练时间 | 模型质量 | 泛化性能 | 推荐场景 |
|------|----------|----------|----------|----------|
| 原始顺序分割 | 1x | 基准 | 可能过拟合 | 不推荐 |
| 改进随机分割 | 1x | 提升 | 更好 | 快速验证 |
| K-Fold交叉验证 | 5x | 最佳 | 最稳定 | 生产部署 |

## 注意事项

1. **内存使用**：K-Fold训练需要额外的磁盘空间存储多个模型
2. **训练时间**：K-Fold训练时间约为单次训练的K倍
3. **模型选择**：K-Fold自动选择验证损失最低的模型作为最佳模型
4. **数据分割**：所有方法都确保同一图像的样本不会跨越训练/验证集

## 故障排查

### 常见问题

1. **内存不足**
   ```bash
   # 减小batch_size
   python finetune_omniparser_lora_kfold.py --batch_size 8
   ```

2. **训练时间过长**
   ```bash
   # 减少epochs或使用更少的folds
   python finetune_omniparser_lora_kfold.py --k_folds 3 --epochs 10
   ```

3. **模型性能不佳**
   - 检查数据质量和标注准确性
   - 尝试调整学习率和LoRA参数
   - 增加训练epochs数量

### 日志分析

训练过程中会输出详细日志，包括：
- 数据分割统计
- 每个fold的训练进度
- 验证损失变化
- 早停机制触发情况
- 最终结果分析

这些信息有助于理解训练过程和优化参数设置。