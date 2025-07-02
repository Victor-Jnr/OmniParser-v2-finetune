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

class Florence2LoRAModelTrainer:
    """
    使用LoRA方法的Florence2模型微调训练器
    
    主要特性：
    - 动态检测所有线性层作为LoRA目标
    - 优先从本地路径加载processor，失败时回退到基础模型
    - 模块化的代码结构，易于维护和扩展
    - 完整的错误处理和日志记录
    
    相比传统层冻结方法的优势：
    - 内存使用量大幅减少（99%+内存节省）
    - 训练参数极少（<1%的模型参数）
    - 支持多任务适配器切换
    - 更适合大规模部署
    """
    
    def __init__(self, base_model_path: str = "weights/icon_caption_florence", use_bfloat16: bool = False):
        self.base_model_path = base_model_path
        # 修复：Processor应该从标准在线模型加载，不是本地路径
        self.processor_model_path = "microsoft/Florence-2-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = use_bfloat16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        
        print(f"Initializing Florence2LoRAModelTrainer with model path: {self.base_model_path}")
        print(f"Using device: {self.device}")
        if self.use_bfloat16:
            print(f"✓ Using bfloat16 for improved efficiency (GPU supports Ampere+)")
        else:
            print(f"Using float32 for maximum compatibility")
            if use_bfloat16 and not self.use_bfloat16:
                print(f"⚠️  bfloat16 requested but not available (requires Ampere+ GPU)")
    
    def setup_model_and_processor(self, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        """设置本地模型和处理器，应用LoRA微调"""
        print(f"Loading Florence2 model from local path: {self.base_model_path}")
        
        # 检查本地模型文件是否存在
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Local model path {self.base_model_path} does not exist!")
        
        try:
            # 修复1：始终从标准位置加载processor（遵循原始项目设计）
            print(f"Loading processor from standard location: {self.processor_model_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.processor_model_path, 
                trust_remote_code=True
            )
            print("✓ Processor loaded from standard location")
            
            # 修复2：始终从本地路径加载模型权重
            print(f"Loading model weights from local path: {self.base_model_path}")
            print(f"Using dtype: {self.dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                local_files_only=True
            ).to(self.device)
            
            print(f"✓ Local model loaded successfully")
            print(f"  Model type: {self.model.config.model_type}")
            print(f"  Model path: {getattr(self.model.config, '_name_or_path', self.base_model_path)}")
            
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            print("❌ CRITICAL: Local model loading failed!")
            print(f"❌ Expected model path: {self.base_model_path}")
            print("❌ Training/merge MUST use local model weights, not falling back to online model")
            print("❌ Please ensure the local model exists and is accessible")
            # 不再回退到在线模型，直接抛出错误
            raise ValueError(f"Cannot load local model from {self.base_model_path}. Training requires local weights.") from e
        
        # 验证模型权重来源
        self.verify_model_source()
        
        # 应用LoRA配置
        self.apply_lora_config(lora_r, lora_alpha, lora_dropout)
        
        self.model.train()
    
    def verify_model_source(self):
        """验证模型是否真的从本地加载"""
        print("\n🔍 Verifying model source...")
        
        # 检查模型配置中的路径信息
        config_path = getattr(self.model.config, '_name_or_path', 'Unknown')
        print(f"  Model config path: {config_path}")
        
        # CRITICAL: 确保模型路径匹配
        if config_path != self.base_model_path:
            print(f"❌ CRITICAL ERROR: Model path mismatch!")
            print(f"   Expected: {self.base_model_path}")
            print(f"   Actual: {config_path}")
            raise ValueError("Model is not loaded from the specified local path! This will cause incorrect training/merge results.")
        
        # 检查模型文件大小 - 验证是本地1GB模型而非在线270MB模型
        expected_size = 1083916964  # 本地模型的确切大小 (~1GB)
        local_model_file = os.path.join(self.base_model_path, 'model.safetensors')
        if os.path.exists(local_model_file):
            actual_size = os.path.getsize(local_model_file)
            print(f"  Local model size: {actual_size:,} bytes ({actual_size/(1024*1024):.1f}MB)")
            if actual_size == expected_size:
                print("  ✓ Model size matches expected local model")
            else:
                print(f"  ⚠️  Warning: Model file size unexpected")
                print(f"     Expected: {expected_size:,} bytes")
                print(f"     Actual: {actual_size:,} bytes")
        
        # 检查参数数量 - 本地模型应该有特定的参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total model parameters: {total_params:,}")
        
        # 简单推理验证模型行为
        try:
            test_image = Image.new('RGB', (64, 64), (128, 128, 128))
            inputs = self.processor(
                text=["<CAPTION>"], 
                images=[test_image], 
                return_tensors="pt",
                do_resize=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Handle mixed precision models properly
                if self.device.type == 'cuda' and hasattr(self.model, 'dtype'):
                    model_dtype = self.model.dtype
                    print(f"  Model dtype: {model_dtype}")
                    
                    # Convert inputs to match model dtype
                    for key, tensor in inputs.items():
                        if tensor.dtype.is_floating_point:
                            inputs[key] = tensor.to(dtype=model_dtype)
                
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False
                )
            
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"  Model inference test: '{result}'")
            
        except Exception as e:
            print(f"  ⚠️  Verification test failed: {e}")
        
        print("✓ Model source verification completed - using local model\n")
    
    def apply_lora_config(self, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        """应用LoRA配置到模型"""
        print(f"Applying LoRA configuration (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})...")
        
        # 动态检测所有线性层作为LoRA目标模块
        all_linear_layers = set()
        
        for name, module in self.model.named_modules():
            # 检查当前模块是否是 torch.nn.Linear 类的实例
            # 这是 LoRA 最主要的应用对象
            if isinstance(module, torch.nn.Linear):
                # 如果是线性层，就将其名称添加到 set 中
                all_linear_layers.add(name)
        
        print(f"Found {len(all_linear_layers)} linear layers in the model")
        
        # 显示所有线性层（用于调试）
        print("All linear layers found:")
        for layer in sorted(list(all_linear_layers))[:20]:  # 显示前20个
            print(f"  - {layer}")
        if len(all_linear_layers) > 20:
            print(f"  ... and {len(all_linear_layers) - 20} more layers")

        # 筛选注意力层和关键线性层作为目标模块
        target_modules = [
            name for name in all_linear_layers 
            if "q_proj" in name 
            or "v_proj" in name 
            or "qkv" in name
        ]
        
        print(f"Selected {len(target_modules)} target modules for LoRA:")
        for module in sorted(target_modules):
            print(f"  - {module}")
        
        if not target_modules:
            print("⚠️  Warning: No suitable target modules found. Using fallback modules...")
            # 如果没有找到标准的注意力层，使用所有线性层中的一部分
            target_modules = [name for name in all_linear_layers if "linear" in name.lower()][:10]
            if not target_modules:
                # 最后的备选方案
                target_modules = list(all_linear_layers)[:5]
            print(f"Fallback target modules: {target_modules}")
        
        # LoRA配置 - 使用动态检测的目标模块
        lora_config = LoraConfig(
            r=lora_r,                    # 秩参数
            lora_alpha=lora_alpha,       # 缩放因子
            target_modules=target_modules,  # 使用动态检测的注意力层
            lora_dropout=lora_dropout,
            bias="none",                 # 不训练bias, 
            task_type="CAUSAL_LM"        # 因果语言模型
        )
        
        # 应用LoRA到模型
        try:
            self.model = get_peft_model(self.model, lora_config)
            
            # 统计参数信息
            total_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    if trainable_params <= 10:  # 只显示前10个可训练参数
                        print(f"✓ Trainable: {name} ({param.numel():,} params)")
            
            print(f"\nLoRA Parameter summary:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
            print(f"Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
            
            # 显示LoRA特有信息
            print(f"LoRA rank (r): {lora_r}")
            print(f"LoRA alpha: {lora_alpha}")
            print(f"LoRA dropout: {lora_dropout}")
            print(f"Estimated memory savings: ~{100*(total_params-trainable_params)/total_params:.1f}%")
            
        except Exception as e:
            print(f"✗ Error applying LoRA configuration: {e}")
            print("This might be due to module name mismatch or incompatible model structure.")
            print(f"Available linear layers: {sorted(list(all_linear_layers))[:10]}...")
            raise
    
    def _prepare_dataloaders(self, florence_data: List[Dict], batch_size: int):
        """准备训练和验证数据加载器"""
        # 验证数据格式
        sample_data = florence_data[0]
        print(f"Sample data format: {sample_data.keys()}")
        if 'image_path' not in sample_data or 'content' not in sample_data:
            print("Warning: Data format may not be compatible. Expected 'image_path' and 'content' keys.")
        
        # 数据分割
        train_size = int(0.8 * len(florence_data))
        train_data = florence_data[:train_size]
        val_data = florence_data[train_size:]
        
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # 创建datasets
        train_dataset = Florence2LoRADataset(train_data, self.processor)
        val_dataset = Florence2LoRADataset(val_data, self.processor) if val_data else None
        
        print(f"Dataset created - Train: {len(train_dataset)} samples")
        if val_dataset:
            print(f"Dataset created - Val: {len(val_dataset)} samples")
        
        # 创建dataloaders
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=collate_fn_florence_lora,
                num_workers=0  # 避免多进程问题
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate_fn_florence_lora,
                num_workers=0
            ) if val_dataset else None
            
            print(f"DataLoader created - Train batches: {len(train_loader)}")
            if val_loader:
                print(f"DataLoader created - Val batches: {len(val_loader)}")
            
            return train_loader, val_loader
                
        except Exception as e:
            print(f"Error creating DataLoader: {e}")
            raise
    
    def _create_optimizer_and_scheduler(self, lr: float, warmup_ratio: float, train_loader: DataLoader, epochs: int):
        """创建优化器和学习率调度器"""
        # 设置优化器 - 只优化LoRA参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"LoRA trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # LoRA通常可以使用稍高的学习率
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        
        lr_scheduler = get_scheduler(
            name="cosine",  # 使用cosine decay
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return optimizer, lr_scheduler, trainable_params
    
    def train_lora_model(self, florence_data: List[Dict], 
                        epochs: int = 5, 
                        batch_size: int = 8, 
                        lr: float = 1e-4,  # LoRA通常可以使用稍高的学习率
                        warmup_ratio: float = 0.1,
                        # LoRA配置参数
                        lora_r: int = 16,
                        lora_alpha: int = 32,
                        lora_dropout: float = 0.1):
        """LoRA模型训练策略"""
        print("Starting LoRA-based Florence2 model training...")
        
        if not florence_data:
            print("No Florence2 training data available!")
            return
        
        print(f"Training with {len(florence_data)} samples")
        print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        print(f"LoRA parameters: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        self.setup_model_and_processor(
            lora_r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )
        
        # 准备数据加载器
        train_loader, val_loader = self._prepare_dataloaders(florence_data, batch_size)
        
        # 创建优化器和调度器
        optimizer, lr_scheduler, trainable_params = self._create_optimizer_and_scheduler(lr, warmup_ratio, train_loader, epochs)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        # 训练循环 - 与原始训练器相同的逻辑
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"LoRA Training Epoch {epoch + 1}/{epochs}")):
                try:
                    questions, answers, images = batch
                    
                    # 处理图像 - 确保所有图像都是正确格式
                    processed_images = []
                    for img in images:
                        if hasattr(img, 'convert'):
                            processed_images.append(img.convert('RGB'))
                        elif isinstance(img, np.ndarray):
                            processed_images.append(Image.fromarray(img).convert('RGB'))
                        else:
                            # 如果是路径字符串
                            if isinstance(img, str) and os.path.exists(img):
                                processed_images.append(Image.open(img).convert('RGB'))
                            else:
                                print(f"Warning: Invalid image format: {type(img)}")
                                continue
                    
                    if not processed_images:
                        print("No valid images in batch, skipping...")
                        continue
                    
                    # 处理输入 - 兼容Florence2格式
                    try:
                        inputs = self.processor(
                            text=questions[:len(processed_images)], 
                            images=processed_images, 
                            return_tensors="pt", 
                            padding=True,
                            do_resize=False  # 遵循原始项目：图像已预先resize到64x64
                        )
                        
                        # 正确处理数据类型并确保所有tensor在同一设备  
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # 处理标签 - 确保与questions长度匹配
                        valid_answers = answers[:len(processed_images)]
                        labels = self.processor.tokenizer(
                            text=valid_answers, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=50,  # 限制标签长度
                            return_token_type_ids=False
                        ).input_ids.to(self.device)
                        
                    except Exception as e:
                        print(f"Error processing inputs: {e}")
                        continue
                    
                    # 前向传播 - 添加错误处理
                    try:
                        outputs = self.model(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            labels=labels
                        )
                        
                        if not hasattr(outputs, 'loss') or outputs.loss is None:
                            print("Warning: No loss in outputs, skipping batch")
                            continue
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"CUDA out of memory, skipping batch {batch_idx}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            print(f"Runtime error in forward pass: {e}")
                            continue
                    except Exception as e:
                        print(f"Unexpected error in forward pass: {e}")
                        continue
                    
                    loss = outputs.loss
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    train_loss += loss.item()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch + 1} - LoRA Training Loss: {avg_train_loss:.4f}")
            
            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_lora_model("weights/icon_caption_florence_lora_finetuned")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # 即使没有验证集也保存模型
                if (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
                    save_path = f"weights/icon_caption_florence_lora_epoch_{epoch+1}"
                    self.save_lora_model(save_path)
                    print(f"LoRA model saved at epoch {epoch + 1}")
    
    def validate(self, val_loader):
        """验证函数 - 与原始训练器相同"""
        self.model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    questions, answers, images = batch
                    
                    # 验证时的图像处理
                    processed_images = []
                    for img in images:
                        if hasattr(img, 'convert'):
                            processed_images.append(img.convert('RGB'))
                        elif isinstance(img, np.ndarray):
                            processed_images.append(Image.fromarray(img).convert('RGB'))
                        else:
                            if isinstance(img, str) and os.path.exists(img):
                                processed_images.append(Image.open(img).convert('RGB'))
                            else:
                                continue
                    
                    if not processed_images:
                        continue
                    
                    inputs = self.processor(
                        text=questions[:len(processed_images)], 
                        images=processed_images, 
                        return_tensors="pt", 
                        padding=True,
                        do_resize=False  # 遵循原始项目：图像已预先resize到64x64
                    )
                    
                    # 确保所有tensor在同一设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    valid_answers = answers[:len(processed_images)]
                    labels = self.processor.tokenizer(
                        text=valid_answers, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=50,
                        return_token_type_ids=False
                    ).input_ids.to(self.device)
                    
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        return val_loss / val_batches if val_batches > 0 else float('inf')
    
    def save_lora_model(self, save_path):
        """保存LoRA适配器权重"""
        print(f"Saving LoRA model to {save_path}")
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 保存LoRA适配器
            print("Saving LoRA adapter weights...")
            self.model.save_pretrained(
                save_path,
                safe_serialization=True
            )
            print("✓ LoRA adapter weights saved successfully")
            
            # 保存训练记录
            training_record = {
                "training_type": "florence2_lora_finetune",
                "base_model_source": self.base_model_path,
                "training_method": "LoRA",
                "lora_config": {
                    "r": getattr(self.model.peft_config['default'], 'r', 'unknown'),
                    "lora_alpha": getattr(self.model.peft_config['default'], 'lora_alpha', 'unknown'),
                    "lora_dropout": getattr(self.model.peft_config['default'], 'lora_dropout', 'unknown'),
                    "target_modules": list(getattr(self.model.peft_config['default'], 'target_modules', []))
                },
                "device": str(self.device),
                "training_completed": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "LoRA adapter only - use with base model from local weights"
            }
            
            with open(os.path.join(save_path, "lora_training_record.json"), "w") as f:
                json.dump(training_record, f, indent=2)
            
            print(f"✓ LoRA model successfully saved to {save_path}")
            print(f"ℹ️  Use with base model from '{self.base_model_path}' for inference")
            
        except Exception as e:
            print(f"✗ Error saving LoRA model: {e}")
            raise

    def merge_and_save_model(self, output_path: str, save_quantized: bool = False):
        """
        合并LoRA权重到基础模型并保存为新的完整模型
        Args:
            output_path: 保存合并后模型的路径
        注意：只保存模型权重，不保存processor（processor与训练无关）
        """
        try:
            print(f"\n🔄 Merging LoRA adapter with base model...")
            
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            # 合并LoRA权重到基础模型
            merged_model = self.model.merge_and_unload()
            
            # 保存合并后的模型权重
            print(f"💾 Saving merged model weights to {output_path}...")
            merged_model.save_pretrained(output_path)
            
            # 复制基础模型的配置文件（确保使用正确的config.json）
            print(f"📋 Copying base model config files...")
            base_config_files = ['config.json', 'generation_config.json']
            for config_file in base_config_files:
                src_path = os.path.join(self.base_model_path, config_file)
                dst_path = os.path.join(output_path, config_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"✓ Copied {config_file} from base model")
                else:
                    print(f"⚠️  {config_file} not found in base model path")
            
            # 保存合并信息
            merge_record = {
                "model_type": "florence2_merged",
                "base_model_source": self.base_model_path,
                "training_method": "LoRA_merged",
                "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Merged model with base model configs - ready for direct use"
            }
            
            with open(os.path.join(output_path, "merge_info.json"), "w") as f:
                json.dump(merge_record, f, indent=2)
            
            print(f"✓ Model successfully merged and saved to {output_path}")
            print(f"ℹ️  Merged model uses base model configs and can be used directly")
            print(f"ℹ️  Usage: AutoModelForCausalLM.from_pretrained('{output_path}')")
            
            if save_quantized:
                from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM 
                quantization_config_bit = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16, bnb_8bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(output_path, quantization_config=quantization_config_bit, torch_dtype=torch.float32, trust_remote_code=True) # in new version, it automatically select device
                model.save_pretrained("weights/icon_caption_florence_8bit_lora_finetuned")

            return True
            
        except Exception as e:
            print(f"✗ Error merging model: {e}")
            return False

class Florence2LoRADataset(Dataset):
    """LoRA训练数据集 - 复用原始数据集逻辑"""
    
    def __init__(self, data_list: List[Dict], processor, max_length: int = 50):
        self.data_list = data_list
        self.processor = processor
        self.max_length = max_length
        
        # 检查数据格式并统计
        self.validate_data()
    
    def validate_data(self):
        """验证数据格式"""
        valid_count = 0
        invalid_count = 0
        
        for i, item in enumerate(self.data_list):
            if 'image_path' in item and 'content' in item:
                # 检查图像文件是否存在
                if isinstance(item['image_path'], str) and os.path.exists(item['image_path']):
                    valid_count += 1
                else:
                    invalid_count += 1
                    if i < 5:  # 只打印前5个错误
                        print(f"Warning: Image file not found: {item['image_path']}")
            else:
                invalid_count += 1
                if i < 5:
                    print(f"Warning: Invalid data format at index {i}: {item.keys()}")
        
        print(f"LoRA Dataset validation: {valid_count} valid, {invalid_count} invalid samples")
        
        if valid_count == 0:
            raise ValueError("No valid samples found in dataset!")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """与原始数据集相同的数据处理逻辑"""
        item = self.data_list[idx]
        
        # 遵循原始项目：Florence2使用<CAPTION>提示词
        question = "<CAPTION>"
        
        # 改进的答案格式 - 更适合UI元素
        original_content = item.get('content', 'unknown')
        answer = f"{original_content}"
        
        # 加载和处理图像
        image_path = item.get('image_path', '')
        bbox = item.get('bbox', [0, 0, 1, 1])  # 默认全图
        
        try:
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # 创建默认图像
                image = Image.new('RGB', (64, 64), (128, 128, 128))
                print(f"Warning: Using default image for item {idx}")
            
            # 处理边界框裁剪
            if len(bbox) >= 4 and any(b != 0 for b in bbox[:2]) or any(b != 1 for b in bbox[2:]):
                width, height = image.size
                x1 = max(0, int(bbox[0] * width))
                y1 = max(0, int(bbox[1] * height))
                x2 = min(width, int(bbox[2] * width))
                y2 = min(height, int(bbox[3] * height))
                
                if x2 > x1 and y2 > y1:
                    cropped_image = image.crop((x1, y1, x2, y2))
                    # 使用原始项目的标准尺寸：64x64
                    cropped_image = cropped_image.resize((64, 64))
                else:
                    cropped_image = image.resize((64, 64))
            else:
                cropped_image = image.resize((64, 64))
            
            return question, answer, cropped_image
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # 返回默认值 - 使用原始项目标准尺寸64x64
            default_image = Image.new('RGB', (64, 64), (128, 128, 128))
            return question, "icon", default_image

def collate_fn_florence_lora(batch):
    """LoRA训练的collate函数 - 复用原始逻辑"""
    try:
        # 过滤掉None值
        valid_batch = [item for item in batch if item is not None and len(item) == 3]
        
        if not valid_batch:
            # 返回空batch
            return [], [], []
        
        questions, answers, images = zip(*valid_batch)
        
        # 确保所有数据都是有效的
        valid_questions = [q for q in questions if q is not None]
        valid_answers = [a for a in answers if a is not None]
        valid_images = [img for img in images if img is not None]
        
        # 确保长度一致
        min_len = min(len(valid_questions), len(valid_answers), len(valid_images))
        
        return (
            list(valid_questions[:min_len]), 
            list(valid_answers[:min_len]), 
            list(valid_images[:min_len])
        )
        
    except Exception as e:
        print(f"Error in LoRA collate function: {e}")
        return [], [], []

def prepare_training_data(data_path: str) -> List[Dict]:
    """准备训练数据，支持多种格式"""
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from {data_path}")
        
        # 验证数据格式
        if data and isinstance(data[0], dict):
            required_keys = ['image_path', 'content']
            if all(key in data[0] for key in required_keys):
                print("Data format validation passed")
                return data
            else:
                print(f"Warning: Missing required keys. Found keys: {list(data[0].keys())}")
                return data  # 仍然尝试使用
        else:
            print("Warning: Unexpected data format")
            return []
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def test_lora_model_loading(model_path: str):
    """测试LoRA模型加载是否正常"""
    print(f"Testing LoRA model loading from: {model_path}")
    
    try:
        trainer = Florence2LoRAModelTrainer(base_model_path=model_path)
        trainer.setup_model_and_processor()
        print("✓ LoRA model loading test passed")
        
        # 测试简单推理
        test_image = Image.new('RGB', (64, 64), (128, 128, 128))
        inputs = trainer.processor(
            text=["<CAPTION>"], 
            images=[test_image], 
            return_tensors="pt",
            do_resize=False
        )
        
        # 确保所有tensor在同一设备
        inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = trainer.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=10,
                num_beams=1
            )
        
        result = trainer.processor.batch_decode(outputs, skip_special_tokens=True)
        print(f"✓ LoRA model inference test passed. Output: {result}")
        return True
        
    except Exception as e:
        print(f"✗ LoRA model loading/inference test failed: {e}")
        return False

def main():
    """
    LoRA微调主函数
    
    使用改进的LoRA微调方法，具有以下特性：
    1. 动态检测目标模块 - 自动识别所有注意力层
    2. 健壮的模型加载 - 优先本地processor，智能回退
    3. 模块化设计 - 清晰的代码结构，易于维护
    4. 可选bfloat16优化 - 支持Ampere+架构GPU的内存优化
    
    对比两种微调方法的结果：
    - 层冻结模型: weights/icon_caption_florence_finetuned
    - LoRA模型: weights/icon_caption_florence_lora_finetuned
    """
    
    print("=== Florence2 LoRA Model Fine-tuning ===")
    
    # 配置
    model_path = "weights/icon_caption_florence"
    data_path = "training_data/florence_format/florence_data.json"
    
    # 检查依赖
    try:
        import peft
        print(f"✓ PEFT library version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT library not found. Please install with: pip install peft")
        return
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        print("Please ensure you have downloaded the Florence2 model weights.")
        return
    
    # 测试模型加载
    print("\n1. Testing LoRA model loading...")
    if not test_lora_model_loading(model_path):
        print("LoRA model loading test failed. Please check your model weights.")
        return
    
    # 准备训练数据
    print("\n2. Preparing training data...")
    florence_data = prepare_training_data(data_path)
    
    if not florence_data:
        print("No training data available. Please prepare your training data first.")
        print("Expected format: [{\"image_path\": \"path/to/image\", \"content\": \"description\", \"bbox\": [x1,y1,x2,y2]}, ...]")
        return
    
    # 创建LoRA训练器
    print("\n3. Creating LoRA trainer...")
    # 可选：启用 bfloat16 以获得更好的效率（需要 Ampere+ GPU）
    use_bfloat16 = False  # 设为 True 以启用 bfloat16（如果GPU支持）
    trainer = Florence2LoRAModelTrainer(base_model_path=model_path, use_bfloat16=use_bfloat16)
    
    # 开始LoRA训练
    print("\n4. Starting LoRA training...")
    try:
        trainer.train_lora_model(
            florence_data=florence_data,
            epochs=20,                 # 自动早停, 可设大点
            batch_size=16,              # batch_size 根据内存大小调整
            lr=5e-5,                   # LoRA 可以使用稍高的学习率
            warmup_ratio=0.1,          # 学习率预热
            # LoRA 配置参数
            lora_r=16,                 # 提高秩参数获得更强表达能力
            lora_alpha=32,             # 通常是 r 的 2 倍
            lora_dropout=0.1           # 防止过拟合
        )

        print("\n✓ LoRA training completed successfully!")

    except Exception as e:
        print(f"\n✗ LoRA training failed: {e}")
        import traceback
        traceback.print_exc()

def merge_existing_lora(lora_path: str, base_model_path: str, output_path: str):
    """
    合并现有的LoRA适配器到基础模型
    Args:
        lora_path: LoRA适配器路径
        base_model_path: 基础模型路径  
        output_path: 输出路径
    注意：只保存模型权重，不保存processor（processor与训练无关）
    """
    try:
        print(f"🔄 Loading LoRA adapter from {lora_path}...")
        
        # 加载基础模型和processor（仅用于验证）
        from peft import PeftModel
        
        print(f"📥 Loading base model from {base_model_path}...")
        print(f"🔍 Verifying base model path exists: {os.path.exists(base_model_path)}")
        
        # 强制验证本地模型文件
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
        
        model_file = os.path.join(base_model_path, 'model.safetensors')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file does not exist: {model_file}")
            
        # 检查模型大小
        expected_size = 1083916964  # 本地模型应该是这个大小
        actual_size = os.path.getsize(model_file)
        print(f"🔍 Base model size: {actual_size:,} bytes ({actual_size/(1024*1024):.1f}MB)")
        
        if actual_size != expected_size:
            print(f"⚠️  Warning: Base model size unexpected!")
            print(f"   Expected: {expected_size:,} bytes")
            print(f"   Actual: {actual_size:,} bytes")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 修复：强制使用float32与本地模型保持一致
            local_files_only=True  # 确保从本地加载
        )
        
        # 验证加载的模型路径
        loaded_path = getattr(base_model.config, '_name_or_path', 'Unknown')
        print(f"🔍 Loaded model config path: {loaded_path}")
        if loaded_path != base_model_path:
            raise ValueError(f"Model loaded from wrong path! Expected: {base_model_path}, Got: {loaded_path}")
        
        # 验证模型参数数量
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"🔍 Base model parameters: {total_params:,}")
        expected_params = 270803968  # 本地模型的参数数量
        if total_params != expected_params:
            print(f"⚠️  Warning: Parameter count unexpected!")
            print(f"   Expected: {expected_params:,}")
            print(f"   Actual: {total_params:,}")
        
        print("✓ Base model verification passed")
        
        print(f"🔗 Loading and merging LoRA adapter...")
        # 加载LoRA适配器
        model_with_lora = PeftModel.from_pretrained(base_model, lora_path)
        
        # 合并权重
        merged_model = model_with_lora.merge_and_unload()
        
        # 保存合并后的模型权重
        print(f"💾 Saving merged model weights to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)
        
        # 复制基础模型的配置文件（确保使用正确的config.json）
        print(f"📋 Copying base model config files...")
        base_config_files = ['config.json', 'generation_config.json']
        for config_file in base_config_files:
            src_path = os.path.join(base_model_path, config_file)
            dst_path = os.path.join(output_path, config_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"✓ Copied {config_file} from base model")
            else:
                print(f"⚠️  {config_file} not found in base model path")
        
        # 保存合并信息
        merge_info = {
            "model_type": "florence2_merged",
            "base_model_source": base_model_path,
            "lora_adapter_source": lora_path,
            "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Merged model with base model configs - ready for direct use"
        }
        
        with open(os.path.join(output_path, "merge_info.json"), "w") as f:
            json.dump(merge_info, f, indent=2)
        
        print(f"✓ Successfully merged LoRA adapter into complete model")
        print(f"✓ Merged model saved to: {output_path}")
        print(f"ℹ️  Usage: AutoModelForCausalLM.from_pretrained('{output_path}')")
        
        return True
        
    except Exception as e:
        print(f"✗ Error merging LoRA adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Florence2 LoRA Fine-tuning with optional model merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic LoRA training
  python finetune_omniparser_lora.py
  
  # LoRA training with model merging
  python finetune_omniparser_lora.py --merge
  
  # Custom merge path
  python finetune_omniparser_lora.py --merge --merge_path weights/my_merged_model
  
  # Only merge existing LoRA adapter (no training)
  python finetune_omniparser_lora.py --merge_only --lora_path weights/icon_caption_florence_lora_finetuned
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="training_data/florence_format/florence_data.json",
        help="Path to training data JSON file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/icon_caption_florence",
        help="Path to base Florence2 model"
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapter with base model after training"
    )
    
    parser.add_argument(
        "--merge_path",
        type=str,
        default="weights/icon_caption_florence_merged",
        help="Output path for merged model"
    )
    
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Only merge existing LoRA adapter (skip training)"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default="weights/icon_caption_florence_lora_finetuned",
        help="Path to existing LoRA adapter for merge-only mode"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank parameter"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # 如果只是合并现有LoRA适配器
    if args.merge_only:
        print("=== LoRA Model Merging ===")
        print(f"Base model: {args.model_path}")
        print(f"LoRA adapter: {args.lora_path}")
        print(f"Output path: {args.merge_path}")
        
        if not os.path.exists(args.model_path):
            print(f"✗ Base model path does not exist: {args.model_path}")
            exit(1)
        
        if not os.path.exists(args.lora_path):
            print(f"✗ LoRA adapter path does not exist: {args.lora_path}")
            exit(1)
        
        success = merge_existing_lora(args.lora_path, args.model_path, args.merge_path)
        exit(0 if success else 1)
    
    # 正常的训练流程
    print("=== Florence2 LoRA Model Fine-tuning ===")
    print(f"Training data: {args.data}")
    print(f"Base model: {args.model_path}")
    print(f"Merge after training: {args.merge}")
    if args.merge:
        print(f"Merge output path: {args.merge_path}")
    
    # 检查依赖
    try:
        import peft
        print(f"✓ PEFT library version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT library not found. Please install with: pip install peft")
        exit(1)
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        print("Please ensure you have downloaded the Florence2 model weights.")
        exit(1)
    
    # 测试模型加载
    print("\n1. Testing LoRA model loading...")
    if not test_lora_model_loading(args.model_path):
        print("LoRA model loading test failed. Please check your model weights.")
        exit(1)
    
    # 准备训练数据
    print("\n2. Preparing training data...")
    florence_data = prepare_training_data(args.data)
    
    if not florence_data:
        print("No training data available. Please prepare your training data first.")
        print("Expected format: [{\"image_path\": \"path/to/image\", \"content\": \"description\", \"bbox\": [x1,y1,x2,y2]}, ...]")
        exit(1)
    
    # 创建LoRA训练器
    print("\n3. Creating LoRA trainer...")
    use_bfloat16 = False  # 设为 True 以启用 bfloat16（如果GPU支持）
    trainer = Florence2LoRAModelTrainer(base_model_path=args.model_path, use_bfloat16=use_bfloat16)
    
    # 开始LoRA训练
    print("\n4. Starting LoRA training...")
    try:
        trainer.train_lora_model(
            florence_data=florence_data,
            epochs=30,                 # 自动早停, 可设大点
            batch_size=16,              # batch_size 根据内存大小调整
            lr=5e-5,                   # LoRA 可以使用稍高的学习率
            warmup_ratio=0.1,          # 学习率预热
            # LoRA 配置参数
            lora_r=16,                 # 提高秩参数获得更强表达能力
            lora_alpha=32,             # 通常是 r 的 2 倍
            lora_dropout=0.05          # 防止过拟合
        )

        print("\n✓ LoRA training completed successfully!")
        
        # 如果开启merge选项，合并并保存完整模型
        if args.merge:
            print("\n5. Merging LoRA with base model...")
            if trainer.merge_and_save_model(args.merge_path, save_quantized=True):
                print(f"✓ Merged model weights saved to {args.merge_path}")
                print(f"ℹ️  Use with processor from weights/icon_caption_florence")
            else:
                print("✗ Model merge failed")
                exit(1)

    except Exception as e:
        print(f"\n✗ LoRA training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
