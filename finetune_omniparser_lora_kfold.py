#!/usr/bin/env python3
"""
Florence2 LoRA K-Fold交叉验证微调训练器

基于改进的 finetune_omniparser_lora.py，实现K-Fold交叉验证训练
通过多次训练获得更稳定和泛化能力更强的模型
"""

import os
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, AutoModelForCausalLM, 
    get_scheduler, Trainer, TrainingArguments
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import cv2
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# 导入原始训练器的组件
from finetune_omniparser_lora import (
    Florence2LoRAModelTrainer, 
    Florence2LoRADataset, 
    collate_fn_florence_lora,
    prepare_training_data
)

class Florence2LoRAKFoldTrainer:
    """
    Florence2 LoRA K-Fold交叉验证训练器
    
    特性：
    - 基于图像的K-Fold分割，避免数据泄露
    - 多轮交叉验证训练，提高模型泛化能力
    - 自动选择最佳模型
    - 支持模型融合
    """
    
    def __init__(self, base_model_path: str = "weights/icon_caption_florence", 
                 k_folds: int = 5, use_bfloat16: bool = False):
        self.base_model_path = base_model_path
        self.k_folds = k_folds
        self.use_bfloat16 = use_bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 存储每个fold的结果
        self.fold_results = []
        self.best_fold = None
        self.best_val_loss = float('inf')
        
        print(f"=== Florence2 LoRA K-Fold Trainer ===")
        print(f"Base model: {self.base_model_path}")
        print(f"K-Fold splits: {self.k_folds}")
        print(f"Device: {self.device}")
    
    def create_image_based_kfold_splits(self, florence_data: List[Dict], seed: int = 42) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        创建基于图像的K-Fold分割
        确保同一图像的样本不会同时出现在训练和验证集中
        """
        print(f"\n=== 创建 {self.k_folds}-Fold 数据分割 ===")
        random.seed(seed)
        
        # 按图像分组
        image_groups = defaultdict(list)
        for item in florence_data:
            image_groups[item['image_path']].append(item)
        
        # 统计信息
        total_images = len(image_groups)
        total_samples = len(florence_data)
        avg_samples_per_image = total_samples / total_images
        
        print(f"数据统计:")
        print(f"  总图像数: {total_images}")
        print(f"  总样本数: {total_samples}")
        print(f"  平均每图像样本数: {avg_samples_per_image:.1f}")
        
        # 验证K值是否合理
        if total_images < self.k_folds:
            print(f"⚠️  警告: 图像数量({total_images})小于K值({self.k_folds})")
            self.k_folds = max(2, total_images // 2)
            print(f"自动调整K值为: {self.k_folds}")
        
        # 随机打乱图像列表
        images = list(image_groups.keys())
        random.shuffle(images)
        
        # 创建K个fold
        fold_size = len(images) // self.k_folds
        remainder = len(images) % self.k_folds
        
        splits = []
        start_idx = 0
        
        for fold in range(self.k_folds):
            # 计算当前fold的大小（处理余数）
            current_fold_size = fold_size + (1 if fold < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            # 验证集图像
            val_images = images[start_idx:end_idx]
            # 训练集图像
            train_images = images[:start_idx] + images[end_idx:]
            
            # 生成训练和验证数据
            train_data = []
            val_data = []
            
            for img in train_images:
                train_data.extend(image_groups[img])
            
            for img in val_images:
                val_data.extend(image_groups[img])
            
            splits.append((train_data, val_data))
            
            print(f"Fold {fold + 1}:")
            print(f"  训练: {len(train_data)} 样本 ({len(train_images)} 图像)")
            print(f"  验证: {len(val_data)} 样本 ({len(val_images)} 图像)")
            
            start_idx = end_idx
        
        return splits
    
    def train_single_fold(self, fold_idx: int, train_data: List[Dict], val_data: List[Dict],
                         epochs: int = 15, batch_size: int = 16, lr: float = 5e-5,
                         lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        """训练单个fold"""
        print(f"\n=== 训练 Fold {fold_idx + 1}/{self.k_folds} ===")
        
        # 创建独立的训练器实例
        trainer = Florence2LoRAModelTrainer(
            base_model_path=self.base_model_path,
            use_bfloat16=self.use_bfloat16
        )
        
        # 设置模型
        trainer.setup_model_and_processor(
            lora_r=lora_r,
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )
        
        # 准备数据加载器
        train_dataset = Florence2LoRADataset(train_data, trainer.processor)
        val_dataset = Florence2LoRADataset(val_data, trainer.processor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_florence_lora,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_florence_lora,
            num_workers=0
        )
        
        # 创建优化器和调度器
        optimizer, lr_scheduler, trainable_params = trainer._create_optimizer_and_scheduler(
            lr, 0.1, train_loader, epochs
        )
        
        # 训练循环
        best_val_loss = float('inf')
        fold_history = []
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            trainer.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}")):
                try:
                    questions, answers, images = batch
                    
                    if not images:
                        continue
                    
                    # 处理图像
                    processed_images = []
                    for img in images:
                        if hasattr(img, 'convert'):
                            processed_images.append(img.convert('RGB'))
                        elif isinstance(img, np.ndarray):
                            processed_images.append(Image.fromarray(img).convert('RGB'))
                        elif isinstance(img, str) and os.path.exists(img):
                            processed_images.append(Image.open(img).convert('RGB'))
                        else:
                            continue
                    
                    if not processed_images:
                        continue
                    
                    # 处理输入
                    inputs = trainer.processor(
                        text=questions[:len(processed_images)],
                        images=processed_images,
                        return_tensors="pt",
                        padding=True,
                        do_resize=False
                    )
                    
                    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                    
                    # 处理标签
                    valid_answers = answers[:len(processed_images)]
                    labels = trainer.processor.tokenizer(
                        text=valid_answers,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=50,
                        return_token_type_ids=False
                    ).input_ids.to(trainer.device)
                    
                    # 前向传播
                    outputs = trainer.model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels
                    )
                    
                    if not hasattr(outputs, 'loss') or outputs.loss is None:
                        continue
                    
                    loss = outputs.loss
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # 验证
            val_loss = trainer.validate(val_loader)
            
            print(f"Fold {fold_idx+1} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                fold_save_path = f"weights/icon_caption_florence_lora_fold_{fold_idx+1}"
                trainer.save_lora_model(fold_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # 记录fold结果
        fold_result = {
            'fold_idx': fold_idx,
            'best_val_loss': best_val_loss,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'history': fold_history,
            'model_path': fold_save_path
        }
        
        self.fold_results.append(fold_result)
        
        # 更新全局最佳模型
        if best_val_loss < self.best_val_loss:
            self.best_val_loss = best_val_loss
            self.best_fold = fold_idx
        
        print(f"Fold {fold_idx+1} 完成，最佳验证损失: {best_val_loss:.4f}")
        
        # 清理内存
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return fold_result
    
    def train_kfold(self, florence_data: List[Dict], 
                   epochs: int = 15, batch_size: int = 16, lr: float = 5e-5,
                   lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
                   seed: int = 42):
        """执行K-Fold交叉验证训练"""
        print(f"\n=== 开始 {self.k_folds}-Fold 交叉验证训练 ===")
        
        # 创建数据分割
        splits = self.create_image_based_kfold_splits(florence_data, seed)
        
        # 训练每个fold
        for fold_idx, (train_data, val_data) in enumerate(splits):
            fold_result = self.train_single_fold(
                fold_idx, train_data, val_data,
                epochs=epochs, batch_size=batch_size, lr=lr,
                lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        
        # 分析结果
        self.analyze_results()
        
        # 保存结果
        self.save_kfold_results()
    
    def analyze_results(self):
        """分析K-Fold训练结果"""
        print(f"\n=== K-Fold 训练结果分析 ===")
        
        val_losses = [result['best_val_loss'] for result in self.fold_results]
        
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        min_val_loss = np.min(val_losses)
        max_val_loss = np.max(val_losses)
        
        print(f"验证损失统计:")
        print(f"  平均值: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"  最小值: {min_val_loss:.4f} (Fold {np.argmin(val_losses) + 1})")
        print(f"  最大值: {max_val_loss:.4f} (Fold {np.argmax(val_losses) + 1})")
        
        print(f"\n各Fold详细结果:")
        for i, result in enumerate(self.fold_results):
            print(f"  Fold {i+1}: {result['best_val_loss']:.4f} "
                  f"({result['train_samples']} train, {result['val_samples']} val)")
        
        print(f"\n🏆 最佳模型: Fold {self.best_fold + 1} (验证损失: {self.best_val_loss:.4f})")
        
        # 模型稳定性评估
        cv_score = std_val_loss / mean_val_loss if mean_val_loss > 0 else 0
        print(f"模型稳定性 (CV): {cv_score:.4f} {'(稳定)' if cv_score < 0.1 else '(不稳定)' if cv_score > 0.2 else '(一般)'}")
    
    def save_kfold_results(self):
        """保存K-Fold训练结果"""
        results_dir = "weights/kfold_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        kfold_summary = {
            'k_folds': self.k_folds,
            'best_fold': self.best_fold,
            'best_val_loss': self.best_val_loss,
            'fold_results': self.fold_results,
            'summary_stats': {
                'mean_val_loss': np.mean([r['best_val_loss'] for r in self.fold_results]),
                'std_val_loss': np.std([r['best_val_loss'] for r in self.fold_results]),
                'min_val_loss': np.min([r['best_val_loss'] for r in self.fold_results]),
                'max_val_loss': np.max([r['best_val_loss'] for r in self.fold_results])
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(results_dir, f"kfold_results_{self.k_folds}fold.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(kfold_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ K-Fold结果已保存到: {results_file}")
        
        # 复制最佳模型
        best_model_path = self.fold_results[self.best_fold]['model_path']
        best_model_copy = "weights/icon_caption_florence_lora_kfold_best"
        
        if os.path.exists(best_model_path):
            if os.path.exists(best_model_copy):
                shutil.rmtree(best_model_copy)
            shutil.copytree(best_model_path, best_model_copy)
            print(f"✓ 最佳模型已复制到: {best_model_copy}")
        
        return results_file
    
    def ensemble_predictions(self, test_data: List[Dict], top_k: int = 3):
        """
        模型集成预测（可选功能）
        使用表现最好的K个模型进行集成预测
        """
        print(f"\n=== 模型集成预测 (Top-{top_k}) ===")
        
        # 选择表现最好的K个模型
        sorted_results = sorted(self.fold_results, key=lambda x: x['best_val_loss'])
        top_models = sorted_results[:top_k]
        
        print(f"选择的模型:")
        for i, result in enumerate(top_models):
            print(f"  {i+1}. Fold {result['fold_idx']+1}: {result['best_val_loss']:.4f}")
        
        # 这里可以实现集成预测逻辑
        # 由于需要加载多个模型，实际实现会比较复杂
        print("集成预测功能待实现...")

def calculate_optimal_k(data_size: int, unique_images: int) -> int:
    """根据数据量计算最优K值"""
    if unique_images >= 80:
        return 5
    elif unique_images >= 50:
        return 4
    elif unique_images >= 30:
        return 3
    else:
        return 2

def main():
    """K-Fold训练主函数"""
    parser = argparse.ArgumentParser(description="Florence2 LoRA K-Fold Cross-Validation Training")
    
    parser.add_argument("--data", type=str, default="training_data/florence_format/florence_data.json")
    parser.add_argument("--model_path", type=str, default="weights/icon_caption_florence")
    parser.add_argument("--k_folds", type=int, default=0, help="K-Fold splits (0 for auto)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import peft
        print(f"✓ PEFT library version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT library not found. Please install with: pip install peft")
        return
    
    # 加载数据
    print("=== 加载训练数据 ===")
    florence_data = prepare_training_data(args.data)
    
    if not florence_data:
        print("No training data available!")
        return
    
    # 计算最优K值
    image_counts = Counter([item['image_path'] for item in florence_data])
    unique_images = len(image_counts)
    
    if args.k_folds == 0:
        k_folds = calculate_optimal_k(len(florence_data), unique_images)
        print(f"自动计算最优K值: {k_folds}")
    else:
        k_folds = args.k_folds
    
    # 创建K-Fold训练器
    trainer = Florence2LoRAKFoldTrainer(
        base_model_path=args.model_path,
        k_folds=k_folds,
        use_bfloat16=False
    )
    
    # 开始K-Fold训练
    try:
        trainer.train_kfold(
            florence_data=florence_data,
            epochs=20,
            batch_size=16,
            lr=5e-5,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=42
        )
        
        print("\n🎉 K-Fold交叉验证训练完成！")
        
    except Exception as e:
        print(f"\n❌ K-Fold训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()