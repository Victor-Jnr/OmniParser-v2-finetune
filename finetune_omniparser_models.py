#!/usr/bin/env python3
"""
OmniParser Models Fine-tuning Script

This script allows you to fine-tune both the YOLO icon detection model and 
the Florence2 icon caption model used in OmniParser.

Usage:
    python finetune_omniparser_models.py --mode yolo --data_dir ./training_data
    python finetune_omniparser_models.py --mode florence2 --data_dir ./training_data
    python finetune_omniparser_models.py --mode both --data_dir ./training_data
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from tqdm import tqdm
import shutil

# YOLO related imports
from ultralytics import YOLO
import cv2

# Florence2 related imports
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class OmniParserDatasetConverter:
    """
    Convert OmniParser output format to training datasets for both YOLO and Florence2
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.yolo_dir = self.data_dir / "yolo_format"
        self.florence_dir = self.data_dir / "florence_format"
        
        # Create directories
        for dir_path in [self.yolo_dir, self.florence_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def convert_omniparser_output_to_training_data(self, parsed_content_list: List[Dict], 
                                                  image_path: str, 
                                                  image_size: Tuple[int, int]):
        """
        Convert OmniParser output format to training data
        
        Args:
            parsed_content_list: Output from get_som_labeled_img()
            image_path: Path to the source image
            image_size: (width, height) of the image
        """
        w, h = image_size
        base_name = Path(image_path).stem
        
        # 1. YOLO format conversion
        yolo_annotations = []
        florence_data = []
        
        for item in parsed_content_list:
            if item['type'] == 'icon' and item['interactivity']:
                bbox = item['bbox']  # [x1, y1, x2, y2] in ratio
                content = item.get('content', '')
                
                # Convert to YOLO format (class_id, center_x, center_y, width, height)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Class 0 for all icons (single class detection)
                yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                
                # Florence2 format (for icon captioning)
                if content and content.strip():
                    florence_data.append({
                        'bbox': bbox,
                        'content': content.strip(),
                        'image_path': image_path
                    })
        
        # Save YOLO annotations
        yolo_txt_path = self.yolo_dir / f"{base_name}.txt"
        with open(yolo_txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # Copy image to YOLO directory
        shutil.copy2(image_path, self.yolo_dir / f"{base_name}.jpg")
        
        return florence_data
    
    def create_yolo_config(self, train_ratio: float = 0.8):
        """Create YOLO training configuration"""
        images = list(self.yolo_dir.glob("*.jpg"))
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        
        # Split dataset
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        # Create train/val directories
        for split, img_list in [("train", train_images), ("val", val_images)]:
            split_dir = self.yolo_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for img_path in img_list:
                # Move image and annotation
                txt_path = img_path.with_suffix('.txt')
                shutil.move(str(img_path), str(split_dir / img_path.name))
                if txt_path.exists():
                    shutil.move(str(txt_path), str(split_dir / txt_path.name))
        
        # Create dataset.yaml
        config = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': 1,  # number of classes
            'names': ['icon']  # class names
        }
        
        with open(self.yolo_dir / "dataset.yaml", 'w') as f:
            yaml.dump(config, f)
        
        return str(self.yolo_dir / "dataset.yaml")

class Florence2IconDataset(Dataset):
    """Dataset for training Florence2 icon captioning"""
    
    def __init__(self, data_list: List[Dict], processor, max_length: int = 512):
        self.data = data_list
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and crop image based on bbox
        image = Image.open(item['image_path']).convert('RGB')
        w, h = image.size
        
        bbox = item['bbox']  # [x1, y1, x2, y2] in ratio
        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        
        # Crop and resize the icon region
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = cropped_image.resize((64, 64))  # Standard size used in OmniParser
        
        # Prepare input for Florence2
        question = "<CAPTION>"  # Florence2 task prompt
        answer = item['content']
        
        return question, answer, cropped_image

def collate_fn_florence(batch):
    """Collate function for Florence2 training"""
    questions, answers, images = zip(*batch)
    return list(questions), list(answers), list(images)

class YOLOTrainer:
    """Trainer for YOLO icon detection model"""
    
    def __init__(self, base_model_path: str = "yolov8n.pt"):
        self.base_model_path = base_model_path
    
    def train(self, data_config_path: str, epochs: int = 100, img_size: int = 640):
        """Train YOLO model"""
        print("Starting YOLO training...")
        
        # Load base model
        model = YOLO(self.base_model_path)
        
        # Train
        results = model.train(
            data=data_config_path,
            epochs=epochs,
            imgsz=img_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch=16,
            workers=4,
            project='yolo_training',
            name='icon_detect',
            save=True,
            plots=True,
            val=True
        )
        
        # Save the best model to weights directory
        best_model_path = model.trainer.best
        target_path = "weights/icon_detect/model.pt"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(best_model_path, target_path)
        
        print(f"YOLO training completed. Best model saved to: {target_path}")
        return results

class Florence2Trainer:
    """Trainer for Florence2 icon captioning model"""
    
    def __init__(self, base_model_path: str = "microsoft/Florence-2-base"):
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_model_and_processor(self):
        """Setup Florence2 model and processor"""
        print("Loading Florence2 model and processor...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        if self.device.type == 'cuda':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
    
    def train(self, florence_data: List[Dict], epochs: int = 5, batch_size: int = 8, lr: float = 1e-5):
        """Train Florence2 model for icon captioning"""
        print("Starting Florence2 training...")
        
        self.setup_model_and_processor()
        
        # Split data
        train_size = int(0.8 * len(florence_data))
        train_data = florence_data[:train_size]
        val_data = florence_data[train_size:]
        
        # Create datasets
        train_dataset = Florence2IconDataset(train_data, self.processor)
        val_dataset = Florence2IconDataset(val_data, self.processor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn_florence
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_florence
        )
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                questions, answers, images = batch
                
                # Process inputs
                inputs = self.processor(
                    text=questions, 
                    images=images, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                # Process labels
                labels = self.processor.tokenizer(
                    text=answers, 
                    return_tensors="pt", 
                    padding=True, 
                    return_token_type_ids=False
                ).input_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                    questions, answers, images = batch
                    
                    inputs = self.processor(
                        text=questions, 
                        images=images, 
                        return_tensors="pt", 
                        padding=True
                    ).to(self.device)
                    
                    labels = self.processor.tokenizer(
                        text=answers, 
                        return_tensors="pt", 
                        padding=True, 
                        return_token_type_ids=False
                    ).input_ids.to(self.device)
                    
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save the trained model
        output_dir = "weights/icon_caption_florence_finetuned"
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print(f"Florence2 training completed. Model saved to: {output_dir}")

def prepare_training_data_from_omniparser_output(data_dir: str, parsed_outputs: List[Dict]):
    """
    Prepare training data from OmniParser outputs
    
    Args:
        data_dir: Directory to save training data
        parsed_outputs: List of dictionaries with format:
            {
                'image_path': str,
                'image_size': tuple,
                'parsed_content_list': list
            }
    """
    converter = OmniParserDatasetConverter(data_dir)
    all_florence_data = []
    
    for output in tqdm(parsed_outputs, desc="Converting data"):
        florence_data = converter.convert_omniparser_output_to_training_data(
            output['parsed_content_list'],
            output['image_path'],
            output['image_size']
        )
        all_florence_data.extend(florence_data)
    
    # Create YOLO config
    yolo_config_path = converter.create_yolo_config()
    
    # Save Florence2 data
    florence_data_path = converter.florence_dir / "florence_data.json"
    with open(florence_data_path, 'w') as f:
        json.dump(all_florence_data, f, indent=2)
    
    return yolo_config_path, all_florence_data

def main():
    parser = argparse.ArgumentParser(description='Fine-tune OmniParser models')
    parser.add_argument('--mode', choices=['yolo', 'florence2', 'both'], required=True,
                       help='Which model(s) to train')
    parser.add_argument('--data_dir', required=True,
                       help='Directory containing training data')
    parser.add_argument('--yolo_epochs', type=int, default=100,
                       help='Number of epochs for YOLO training')
    parser.add_argument('--florence_epochs', type=int, default=5,
                       help='Number of epochs for Florence2 training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for Florence2 training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for Florence2 training')
    
    args = parser.parse_args()
    
    # Example: Load your prepared data
    # You need to prepare parsed_outputs with your actual data
    # Here's the expected format:
    """
    parsed_outputs = [
        {
            'image_path': 'path/to/image1.jpg',
            'image_size': (1920, 1080),
            'parsed_content_list': [
                {
                    'type': 'icon',
                    'bbox': [0.1, 0.1, 0.2, 0.2],
                    'interactivity': True,
                    'content': 'Settings button',
                    'source': 'box_yolo_content_yolo'
                },
                # ... more elements
            ]
        },
        # ... more images
    ]
    """
    
    print("Note: You need to prepare your parsed_outputs data.")
    print("Use OmniParser to process your images and collect the outputs.")
    print("Then modify this script to load your actual data.")
    
    # For demonstration, create dummy data structure
    print("\nExample data preparation from existing OmniParser outputs:")
    print("1. Run OmniParser on your training images")
    print("2. Collect parsed_content_list outputs") 
    print("3. Format as shown in the comments above")
    print("4. Run this script with your prepared data")
    
    # Placeholder for actual training
    if os.path.exists(args.data_dir):
        if args.mode in ['yolo', 'both']:
            yolo_config = os.path.join(args.data_dir, 'yolo_format', 'dataset.yaml')
            if os.path.exists(yolo_config):
                trainer = YOLOTrainer()
                trainer.train(yolo_config, epochs=args.yolo_epochs)
        
        if args.mode in ['florence2', 'both']:
            florence_data_path = os.path.join(args.data_dir, 'florence_format', 'florence_data.json')
            if os.path.exists(florence_data_path):
                with open(florence_data_path, 'r') as f:
                    florence_data = json.load(f)
                
                trainer = Florence2Trainer()
                trainer.train(
                    florence_data, 
                    epochs=args.florence_epochs,
                    batch_size=args.batch_size,
                    lr=args.learning_rate
                )

if __name__ == "__main__":
    main() 