#!/usr/bin/env python3
"""
Data Augmentation Script for Florence Format Training Data
Performs image augmentation to increase training data for fine-tuning.
"""

import json
import os
import cv2
import numpy as np
import random
import shutil
import time
from pathlib import Path
import argparse


def apply_random_rotation(image, max_angle=15):
    """Apply random rotation to image within specified angle range."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, f"rot_{angle:.1f}"


def apply_random_crop(image, max_crop_percent=3):
    """Apply random crop to image within specified percentage range."""
    h, w = image.shape[:2]
    
    # Calculate crop percentage (keep it small to preserve content)
    crop_percent = random.uniform(0.5, max_crop_percent) / 100
    
    # Calculate crop dimensions
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    
    # Random crop position
    start_y = random.randint(0, crop_h)
    start_x = random.randint(0, crop_w)
    end_y = h - random.randint(0, crop_h)
    end_x = w - random.randint(0, crop_w)
    
    # Crop and resize back to original size
    cropped = image[start_y:end_y, start_x:end_x]
    resized = cv2.resize(cropped, (w, h))
    
    return resized, f"crop_{crop_percent:.2f}"


def apply_brightness_adjustment(image, brightness_range=(-20, 20)):
    """Apply random brightness adjustment."""
    brightness_delta = random.randint(brightness_range[0], brightness_range[1])
    
    # Convert to float to avoid overflow
    adjusted = image.astype(np.float32)
    adjusted += brightness_delta
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted, f"bright_{brightness_delta}"


def apply_contrast_adjustment(image, contrast_range=(0.8, 1.2)):
    """Apply random contrast adjustment."""
    contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
    
    # Convert to float and apply contrast
    adjusted = image.astype(np.float32)
    adjusted = adjusted * contrast_factor
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted, f"contrast_{contrast_factor:.2f}"


def apply_gaussian_noise(image, noise_strength=10):
    """Apply random gaussian noise."""
    noise = np.random.normal(0, noise_strength, image.shape).astype(np.float32)
    
    # Add noise to image
    noisy = image.astype(np.float32) + noise
    
    # Clip values to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy, f"noise_{noise_strength}"


def apply_random_augmentation(image):
    """Apply a random combination of augmentations."""
    augmentations = [
        apply_random_rotation,
        apply_random_crop,
        apply_brightness_adjustment,
        apply_contrast_adjustment,
        apply_gaussian_noise
    ]
    
    # Choose 1-2 random augmentations
    num_augs = random.randint(1, 2)
    chosen_augs = random.sample(augmentations, num_augs)
    
    aug_names = []
    result_image = image.copy()
    
    for aug_func in chosen_augs:
        result_image, aug_name = aug_func(result_image)
        aug_names.append(aug_name)
    
    return result_image, "_".join(aug_names)


def crop_bbox_from_image(image, bbox):
    """
    Crop image based on bbox coordinates and return cropped image.
    
    Args:
        image: OpenCV image (BGR format)
        bbox: [x1, y1, x2, y2] in normalized coordinates (0-1)
    
    Returns:
        cropped_image: Cropped image region
    """
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x1 = max(0, int(bbox[0] * w))
    y1 = max(0, int(bbox[1] * h))
    x2 = min(w, int(bbox[2] * w))
    y2 = min(h, int(bbox[3] * h))
    
    # Ensure valid crop region
    if x2 <= x1 or y2 <= y1:
        print(f"Warning: Invalid bbox {bbox}, using full image")
        return image
    
    # Crop the region
    cropped = image[y1:y2, x1:x2]
    
    # Resize to 64x64 (following OmniParser standard)
    cropped_resized = cv2.resize(cropped, (64, 64))
    
    return cropped_resized


def augment_data(data_dir="training_data/florence_format", multiplier=3):
    """
    Augment training data by specified multiplier.
    Crops bbox regions first, then applies augmentation to cropped regions.
    
    Args:
        data_dir: Directory containing florence_data.json
        multiplier: Number of times to multiply the original data
    """
    # Paths
    data_path = Path(data_dir)
    json_path = data_path / "florence_data.json"
    augmented_imgs_dir = data_path / "imgs"
    
    # Create augmented images directory
    augmented_imgs_dir.mkdir(exist_ok=True)
    
    # Load original data
    with open(json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Backup original data
    backup_path = data_path / "florence_data_original.json"
    shutil.copy2(json_path, backup_path)
    print(f"Original data backed up to: {backup_path}")
    
    # Create augmented data
    augmented_data = original_data.copy()
    
    # Process each individual item (not by image, but by bbox)
    for idx, item in enumerate(original_data):
        image_path = item["image_path"]
        bbox = item.get("bbox", [0, 0, 1, 1])
        content = item.get("content", "unknown")
        
        # Load image
        full_image_path = Path(image_path)
        if not full_image_path.exists():
            # Try relative path from project root
            full_image_path = Path("/mnt/e/git/OmniParser") / image_path
        
        if not full_image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = cv2.imread(str(full_image_path))
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        print(f"Processing item {idx + 1}/{len(original_data)}: {content} from {Path(image_path).name}")
        
        # Crop the bbox region from original image
        cropped_image = crop_bbox_from_image(image, bbox)
        
        # Generate augmented versions of the cropped region
        for aug_idx in range(multiplier - 1):  # -1 because we keep the original
            try:
                # Apply augmentation to cropped image
                augmented_image, aug_name = apply_random_augmentation(cropped_image)
                
                # Create new filename for augmented cropped image
                original_name = Path(image_path).stem
                safe_content = "".join(c for c in content if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_content = safe_content.replace(' ', '_')[:20]  # Limit length
                
                new_filename = f"{original_name}_{safe_content}_aug_{aug_idx + 1}_{aug_name}.png"
                new_image_path = augmented_imgs_dir / new_filename
                
                # Save augmented cropped image
                cv2.imwrite(str(new_image_path), augmented_image)
                
                # Create new data entry with full image coordinates [0, 0, 1, 1]
                new_item = {
                    "image_path": f"training_data/florence_format/imgs/{new_filename}",
                    "content": content,
                    "bbox": [0, 0, 1, 1]  # Full image coordinates since it's now a cropped image
                }
                augmented_data.append(new_item)
                
            except Exception as e:
                print(f"Error processing augmentation {aug_idx + 1} for item {idx}: {e}")
                continue
    
    # Save augmented data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nData augmentation complete!")
    print(f"Original data: {len(original_data)} items")
    print(f"Augmented data: {len(augmented_data)} items")
    print(f"Multiplier achieved: {len(augmented_data) / len(original_data):.2f}x")
    print(f"Augmented images saved to: {augmented_imgs_dir}")
    print(f"Updated JSON saved to: {json_path}")


def clean_missing_images(data_dir="training_data/florence_format"):
    """
    Clean up JSON data by removing entries for images that no longer exist.
    Useful after manually deleting unwanted augmented images.
    
    Args:
        data_dir: Directory containing florence_data.json
    """
    # Paths
    data_path = Path(data_dir)
    json_path = data_path / "florence_data.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        return
    
    # Load current data
    with open(json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"Original data entries: {len(original_data)}")
    
    # Filter out entries for missing images
    cleaned_data = []
    removed_count = 0
    
    for item in original_data:
        image_path = item.get("image_path", "")
        
        # Check different possible paths
        full_paths = [
            Path(image_path),
            Path(data_dir).parent.parent / image_path,  # Relative to project root
            Path("/mnt/e/git/OmniParser") / image_path   # Absolute project root
        ]
        
        image_exists = any(path.exists() for path in full_paths)
        
        if image_exists:
            cleaned_data.append(item)
        else:
            removed_count += 1
            print(f"Removed entry for missing image: {image_path}")
    
    # Backup original data before cleaning
    if removed_count > 0:
        backup_path = data_path / f"florence_data_before_cleanup_{int(time.time())}.json"
        shutil.copy2(json_path, backup_path)
        print(f"Original data backed up to: {backup_path}")
        
        # Save cleaned data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCleanup complete!")
        print(f"Removed {removed_count} entries for missing images")
        print(f"Remaining entries: {len(cleaned_data)}")
        print(f"Cleaned data saved to: {json_path}")
    else:
        print("No missing images found. No cleanup needed.")


def main():
    parser = argparse.ArgumentParser(description="Augment Florence format training data")
    parser.add_argument("--data_dir", default="training_data/florence_format", 
                       help="Directory containing florence_data.json")
    parser.add_argument("--multiplier", type=int, default=3,
                       help="Data multiplication factor")
    parser.add_argument("--clean_missing", action="store_true",
                       help="Clean up JSON data by removing entries for missing images")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found!")
        print("Please run this script from the OmniParser root directory.")
        return
    
    if args.clean_missing:
        clean_missing_images(args.data_dir)
    else:
        augment_data(args.data_dir, args.multiplier)


if __name__ == "__main__":
    main()