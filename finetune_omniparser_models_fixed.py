import os
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, AutoModelForCausalLM, AdamW, 
    get_scheduler, Trainer, TrainingArguments
)
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import cv2
import random
import time
from pathlib import Path

class Florence2LocalModelTrainer:
    """兼容本地Florence2模型的微调训练器，解决数据格式和模型加载问题"""
    
    def __init__(self, base_model_path: str = "weights/icon_caption_florence"):
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Florence2LocalModelTrainer with model path: {self.base_model_path}")
        print(f"Using device: {self.device}")
    
    def setup_model_and_processor(self, freeze_backbone=True):
        """设置本地模型和处理器，支持层冻结"""
        print(f"Loading Florence2 model from local path: {self.base_model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Model path {self.base_model_path} does not exist!")
        
        try:
            # 遵循原始项目设计：processor总是从基础模型加载，只有模型权重从本地加载
            print("Loading processor from base Florence2 model (as per original project design)")
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            )
            
            # 加载本地模型权重 - 强制使用float32以避免混合精度问题
            print(f"Loading model weights from local path: {self.base_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,  # 强制使用float32统一精度
                trust_remote_code=True,
                local_files_only=True
            ).to(self.device)
            
            print(f"✓ Local model loaded successfully")
            print(f"  Model type: {self.model.config.model_type}")
            print(f"  Model path: {getattr(self.model.config, '_name_or_path', self.base_model_path)}")
            print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            print("Fallback: Loading base Florence2 model for training...")
            
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            )
            
            # 如果使用基础模型作为备选，也使用float32格式
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
                
            print("⚠️  Using base model instead of local weights!")
        
        # 验证模型权重来源
        self.verify_model_source()
        
        # 添加严格验证：确保我们真的在使用本地模型
        self._strict_local_model_verification()
        
        # 冻结大部分参数，只训练顶层
        if freeze_backbone:
            self.freeze_model_layers()
        
        self.model.train()
    
    def verify_model_source(self):
        """验证模型是否真的从本地加载"""
        print("\n🔍 Verifying model source...")
        
        # 检查模型配置中的路径信息
        config_path = getattr(self.model.config, '_name_or_path', 'Unknown')
        print(f"  Model config path: {config_path}")
        
        # 检查模型文件大小（本地微调模型应该与原始不同）
        model_files = []
        if os.path.exists(self.base_model_path):
            for file in os.listdir(self.base_model_path):
                if file.endswith(('.safetensors', '.bin', '.pt')):
                    file_path = os.path.join(self.base_model_path, file)
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    model_files.append((file, f"{size_mb:.1f}MB"))
        
        if model_files:
            print(f"  Local model files: {model_files}")
        
        # 尝试简单推理验证模型行为
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
                        print(f"  {key} dtype: {inputs[key].dtype}")
                
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False
                )
            
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"  Model inference test: '{result}'")
            
            # 检查输出是否包含UI相关内容（本地微调模型的特征）
            if any(keyword in result.lower() for keyword in ['ui', 'button', 'icon', 'menu', 'tab']):
                print("  ✓ Model appears to be UI-specialized (likely local)")
            elif len(result.strip()) < 20:  # 短输出可能是原始模型
                print("  ⚠️  Short output - could be base model or specialized")
            else:
                print("  ⚠️  Output doesn't seem UI-specialized")
                
        except Exception as e:
            print(f"  ⚠️  Verification test failed: {e}")
        
        print("🔍 Model verification completed\n")
    
    def _strict_local_model_verification(self):
        """严格验证模型是否为本地模型，如果不是则停止训练"""
        print("🔒 Strict local model verification...")
        
        # 检查模型配置路径
        model_path = getattr(self.model.config, '_name_or_path', '')
        if model_path != self.base_model_path:
            print(f"❌ CRITICAL: Model path mismatch!")
            print(f"   Expected: {self.base_model_path}")
            print(f"   Actual: {model_path}")
            raise ValueError("Model is not loaded from specified local path!")
        
        # 检查模型文件大小应该匹配本地模型
        expected_size = 1083916964  # 本地模型的确切大小
        local_model_file = os.path.join(self.base_model_path, 'model.safetensors')
        if os.path.exists(local_model_file):
            actual_size = os.path.getsize(local_model_file)
            if actual_size != expected_size:
                print(f"⚠️  Warning: Model file size mismatch")
                print(f"   Expected: {expected_size:,} bytes")
                print(f"   Actual: {actual_size:,} bytes")
        
        # 进行推理对比验证
        try:
            print("Testing inference signature...")
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
                    # Convert inputs to match model dtype
                    for key, tensor in inputs.items():
                        if tensor.dtype.is_floating_point:
                            inputs[key] = tensor.to(dtype=model_dtype)
                
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False
                )
            
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"   Inference result: '{result}'")
            
            # 检查输出特征：本地模型应该有特定的UI输出模式
            if len(result.strip()) > 50:  # 如果输出过长，可能是基础模型
                print("⚠️  WARNING: Output seems too generic, might be base model")
            
            # 记录当前模型状态用于后续对比
            self._initial_inference_result = result
            print("✓ Strict verification passed")
            
        except Exception as e:
            print(f"⚠️  Verification inference failed: {e}")
        
        print("🔒 Strict verification completed\n")
    
    def freeze_model_layers(self):
        """冻结模型的backbone层，只训练顶层"""
        print("Freezing model backbone...")
        
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 统计模型层信息
        total_params = 0
        trainable_params = 0
        
        # 只解冻最关键的层用于微调
        trainable_keywords = [
            'language_model.lm_head',  # 输出层
            'language_model.model.layers.5',  # 最后一层transformer
            'language_model.model.layers.4',  # 倒数第二层
            'projector'  # 投影层
        ]
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # 检查是否应该训练这个参数
            should_train = any(keyword in name for keyword in trainable_keywords)
            
            if should_train:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"✓ Trainable: {name} ({param.numel():,} params)")
            else:
                param.requires_grad = False
        
        print(f"\nParameter summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
    
    def train_local_model(self, florence_data: List[Dict], 
                         epochs: int = 5, 
                         batch_size: int = 8, 
                         lr: float = 1e-5,  # 适中的学习率
                         warmup_ratio: float = 0.1,
                         save_steps: int = 500,
                         eval_steps: int = 100):
        """改进的本地模型训练策略"""
        print("Starting improved Florence2 local model training...")
        
        if not florence_data:
            print("No Florence2 training data available!")
            return
        
        print(f"Training with {len(florence_data)} samples")
        print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        
        self.setup_model_and_processor(freeze_backbone=True)
        
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
        
        # 创建datasets with improved format handling
        train_dataset = Florence2LocalDataset(train_data, self.processor)
        val_dataset = Florence2LocalDataset(val_data, self.processor) if val_data else None
        
        print(f"Dataset created - Train: {len(train_dataset)} samples")
        if val_dataset:
            print(f"Dataset created - Val: {len(val_dataset)} samples")
        
        # 创建dataloaders with error handling
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=collate_fn_florence_local,
                num_workers=0  # 避免多进程问题
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate_fn_florence_local,
                num_workers=0
            ) if val_dataset else None
            
            print(f"DataLoader created - Train batches: {len(train_loader)}")
            if val_loader:
                print(f"DataLoader created - Val batches: {len(val_loader)}")
                
        except Exception as e:
            print(f"Error creating DataLoader: {e}")
            raise
        
        # 设置优化器 - 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        
        lr_scheduler = get_scheduler(
            name="cosine",  # 使用cosine decay
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
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
                        # 保持所有输入为float32，与模型统一
                        
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
            print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")
            
            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model("weights/icon_caption_florence_finetuned")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # 即使没有验证集也保存模型
                if (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
                    save_path = f"weights/icon_caption_florence_epoch_{epoch+1}"
                    self.save_model(save_path)
                    print(f"Model saved at epoch {epoch + 1}")
    
    def validate(self, val_loader):
        """验证函数"""
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
                    # 保持所有输入为float32，与模型统一
                    
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
    
    def save_model(self, save_path):
        """保存模型权重，保持与原始本地模型的兼容性"""
        print(f"Saving model to {save_path}")
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 使用标准方法保存模型，但限制保存的文件
            print("Saving model using standard method to handle shared tensors properly")
            
            # 使用HuggingFace的标准保存方法处理共享张量
            self.model.save_pretrained(
                save_path,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            print("✓ Model weights saved successfully")
            
            # 删除我们不需要的文件（保持与原始本地模型一致）
            unwanted_files = ['configuration_florence2.py', 'modeling_florence2.py']
            for file_name in unwanted_files:
                file_path = os.path.join(save_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed {file_name} to match original model structure")
            
            # 恢复原始本地模型的config文件（被save_pretrained覆盖的）
            print("Restoring original local model configuration files...")
            essential_files = ['config.json', 'generation_config.json']
            for file_name in essential_files:
                source_file = os.path.join(self.base_model_path, file_name)
                if os.path.exists(source_file):
                    target_file = os.path.join(save_path, file_name)
                    import shutil
                    shutil.copy2(source_file, target_file)
                    print(f"Restored {file_name} from original local model")
            
            # 保存训练记录（区别于模型配置）
            training_record = {
                "training_type": "florence2_local_finetune",
                "base_model_source": self.base_model_path,
                "trained_on": "local_weights",
                "device": str(self.device),
                "training_completed": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Model weights only - use original processor from microsoft/Florence-2-base"
            }
            
            with open(os.path.join(save_path, "training_record.json"), "w") as f:
                json.dump(training_record, f, indent=2)
            
            print(f"✓ Model weights successfully saved to {save_path}")
            print(f"ℹ️  Use with processor from 'microsoft/Florence-2-base' for compatibility")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            raise

class Florence2LocalDataset(Dataset):
    """本地Florence2数据集，支持多种数据格式"""
    
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
        
        print(f"Dataset validation: {valid_count} valid, {invalid_count} invalid samples")
        
        if valid_count == 0:
            raise ValueError("No valid samples found in dataset!")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 遵循原始项目：Florence2使用<CAPTION>提示词，但训练时用questions数组
        question = "<CAPTION>"
        
        # 改进的答案格式 - 更适合UI元素
        original_content = item.get('content', 'unknown')
        
        # 根据内容类型调整答案格式
        if any(word in original_content.lower() for word in ['button', 'icon', 'menu', 'tab']):
            answer = f"{original_content}"
        elif any(word in original_content.lower() for word in ['text', 'label', 'title']):
            answer = f"text: {original_content}"
        else:
            answer = f"UI: {original_content}"
        
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

def collate_fn_florence_local(batch):
    """本地模型训练的collate函数，包含错误处理"""
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
        print(f"Error in collate function: {e}")
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

def test_model_loading(model_path: str):
    """测试模型加载是否正常"""
    print(f"Testing model loading from: {model_path}")
    
    try:
        trainer = Florence2LocalModelTrainer(base_model_path=model_path)
        trainer.setup_model_and_processor(freeze_backbone=False)
        print("✓ Model loading test passed")
        
        # 测试简单推理 - 使用原始项目的64x64尺寸
        test_image = Image.new('RGB', (64, 64), (128, 128, 128))
        inputs = trainer.processor(
            text=["<CAPTION>"], 
            images=[test_image], 
            return_tensors="pt",
            do_resize=False  # 遵循原始项目
        )
        
        # 确保所有tensor在同一设备
        inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Handle mixed precision models properly
            if trainer.device.type == 'cuda' and hasattr(trainer.model, 'dtype'):
                model_dtype = trainer.model.dtype
                # Convert inputs to match model dtype
                for key, tensor in inputs.items():
                    if tensor.dtype.is_floating_point:
                        inputs[key] = tensor.to(dtype=model_dtype)
            
            outputs = trainer.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=10,
                num_beams=1
            )
        
        result = trainer.processor.batch_decode(outputs, skip_special_tokens=True)
        print(f"✓ Model inference test passed. Output: {result}")
        return True
        
    except Exception as e:
        print(f"✗ Model loading/inference test failed: {e}")
        return False

def main():
    """改进的主训练函数"""
    print("=== Florence2 Local Model Fine-tuning ===")
    
    # 配置
    model_path = "weights/icon_caption_florence"
    data_path = "training_data/florence_format/florence_data.json"
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        print("Please ensure you have downloaded the Florence2 model weights.")
        return
    
    # 测试模型加载
    print("\n1. Testing model loading...")
    if not test_model_loading(model_path):
        print("Model loading test failed. Please check your model weights.")
        return
    
    # 准备训练数据
    print("\n2. Preparing training data...")
    florence_data = prepare_training_data(data_path)
    
    if not florence_data:
        print("No training data available. Please prepare your training data first.")
        print("Expected format: [{\"image_path\": \"path/to/image\", \"content\": \"description\", \"bbox\": [x1,y1,x2,y2]}, ...]")
        return
    
    # 创建训练器
    print("\n3. Creating trainer...")
    trainer = Florence2LocalModelTrainer(base_model_path=model_path)
    
    # 开始训练
    print("\n4. Starting training...")
    try:
        trainer.train_local_model(
            florence_data=florence_data,
            epochs=7,
            batch_size=3,  
            lr=1e-7,  
            warmup_ratio=0.1 # 学习率预热是一种训练技巧，用于在训练初期逐渐增加学习率，以帮助模型更快地收敛。
        )
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

