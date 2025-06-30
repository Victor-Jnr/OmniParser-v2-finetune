#!/usr/bin/env python3
"""
Training Data Collection Script for OmniParser

This script processes images using the current OmniParser model to generate
training data for both YOLO icon detection and Florence2 icon captioning.

Usage:
    python collect_training_data.py --input_dir ./raw_images --output_dir ./training_data
    python collect_training_data.py --input_dir ./raw_images --output_dir ./training_data --manual_correction
"""

import os
import json
import argparse
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import shutil

# Import OmniParser components
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from finetune_omniparser_models import prepare_training_data_from_omniparser_output

class TrainingDataCollector:
    """Collect training data using existing OmniParser models"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.som_model = None
        self.caption_model_processor = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize OmniParser models"""
        print("Loading OmniParser models...")
        
        # Load YOLO model
        model_path = 'weights/icon_detect/model.pt'
        if os.path.exists(model_path):
            self.som_model = get_yolo_model(model_path)
            self.som_model.to(self.device)
            print(f"YOLO model loaded from {model_path}")
        else:
            print(f"Warning: YOLO model not found at {model_path}")
        
        # Load Florence2 model
        caption_model_path = 'weights/icon_caption_florence'
        if os.path.exists(caption_model_path):
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2", 
                model_name_or_path=caption_model_path, 
                device=self.device
            )
            print(f"Florence2 model loaded from {caption_model_path}")
        else:
            print(f"Warning: Florence2 model not found at {caption_model_path}")
    
    def process_single_image(self, image_path: str, box_threshold: float = 0.05) -> Dict:
        """Process a single image with OmniParser"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        
        # Configure drawing parameters
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # Perform OCR
        ocr_bbox_rslt, _ = check_ocr_box(
            image_path, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Get SOM labeled image with captions
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path, 
            self.som_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            use_local_semantics=True, 
            iou_threshold=0.7, 
            scale_img=False
        )
        
        return {
            'image_path': image_path,
            'image_size': (w, h),
            'parsed_content_list': parsed_content_list,
            'label_coordinates': label_coordinates,
            'dino_labeled_img': dino_labeled_img
        }
    
    def process_image_directory(self, input_dir: str, output_dir: str, 
                              manual_correction: bool = False,
                              box_threshold: float = 0.05) -> List[Dict]:
        """Process all images in a directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        imgs_output_path = os.path.join(output_path, 'imgs')
        os.makedirs(imgs_output_path, exist_ok=True)
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        all_results = []
        failed_images = []
        
        for image_path in tqdm(image_files, desc="Processing images"):
            original_image_path = image_path  # Keep reference to original path object
            try:
                result = self.process_single_image(str(image_path), box_threshold)
                all_results.append(result)
                
                # Save individual result for inspection
                result_file = output_path / f"{image_path.stem}_result.json"
                with open(result_file, 'w') as f:
                    # Save labeled image to imgs_output_path
                    image_name = image_path.stem + '_labeled.png'
                    labeled_image_path = os.path.join(imgs_output_path, image_name)
                    
                    # Handle different types of dino_labeled_img
                    dino_labeled_img = result['dino_labeled_img']
                    if isinstance(dino_labeled_img, str):
                        # Check if it's a base64 encoded image or file path
                        if dino_labeled_img.startswith(('iVBORw0KGgo', '/9j/', 'data:image')) or len(dino_labeled_img) > 1000:
                            import base64
                            import io
                            try:
                                # Remove data URI prefix if present
                                if dino_labeled_img.startswith('data:image'):
                                    base64_data = dino_labeled_img.split(',')[1]
                                else:
                                    base64_data = dino_labeled_img
                                
                                # Decode base64 to image
                                image_data = base64.b64decode(base64_data)
                                labeled_img = Image.open(io.BytesIO(image_data))
                                labeled_img.save(labeled_image_path)
                            except Exception as e:
                                print(f"Warning: Could not decode base64 image for {image_path.name}: {e}")
                        else:
                            # If it's a file path, load and save to our desired location
                            try:
                                labeled_img = Image.open(dino_labeled_img)
                                labeled_img.save(labeled_image_path)
                            except Exception as e:
                                print(f"Warning: Could not load image from path {dino_labeled_img}: {e}")
                    elif hasattr(dino_labeled_img, 'save'):
                        # If it's a PIL Image, save directly
                        try:
                            dino_labeled_img.save(labeled_image_path)
                        except Exception as e:
                            print(f"Warning: Could not save PIL Image for {image_path.name}: {e}")
                    else:
                        print(f"Warning: Unknown type for dino_labeled_img: {type(dino_labeled_img)}")
                    
                    # Prepare JSON data
                    result_copy = result.copy()
                    if 'dino_labeled_img' in result_copy:
                        result_copy['dino_labeled_img'] = f'saved to {labeled_image_path}'
                    json.dump(result_copy, f, indent=2)
                
                print(f"Processed {original_image_path.name}: {len(result['parsed_content_list'])} elements detected")
                
            except Exception as e:
                print(f"Failed to process {original_image_path.name}: {str(e)}")
                failed_images.append(str(original_image_path))
        
        # Save summary
        summary = {
            'total_images': len(image_files),
            'successful': len(all_results),
            'failed': len(failed_images),
            'failed_images': failed_images
        }
        
        with open(output_path / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nProcessing completed:")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        
        if manual_correction:
            self.create_manual_correction_interface(all_results, output_path)
        
        return all_results
    
    def create_manual_correction_interface(self, results: List[Dict], output_path: Path):
        """Create CSV files for manual correction of annotations"""
        print("\nCreating manual correction interface...")
        
        # Create a CSV for manual review and correction
        correction_data = []
        
        for result in results:
            image_path = result['image_path']
            image_name = Path(image_path).name
            
            for i, item in enumerate(result['parsed_content_list']):
                if item['type'] == 'icon' and item['interactivity']:
                    correction_data.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'element_id': i,
                        'bbox_x1': item['bbox'][0],
                        'bbox_y1': item['bbox'][1], 
                        'bbox_x2': item['bbox'][2],
                        'bbox_y2': item['bbox'][3],
                        'original_content': item.get('content', ''),
                        'corrected_content': item.get('content', ''),  # User can edit this
                        'keep_element': True,  # User can set to False to exclude
                        'source': item['source']
                    })
        
        df = pd.DataFrame(correction_data)
        correction_file = output_path / "manual_correction.csv"
        df.to_csv(correction_file, index=False)
        
        print(f"Manual correction file created: {correction_file}")
        print("Instructions for manual correction:")
        print("1. Open manual_correction.csv in a spreadsheet editor")
        print("2. Review and edit 'corrected_content' column")
        print("3. Set 'keep_element' to False for elements you want to exclude")
        print("4. Save the file and run: python collect_training_data.py --apply_corrections")
    
    def apply_manual_corrections(self, output_dir: str) -> List[Dict]:
        """Apply manual corrections from CSV file"""
        output_path = Path(output_dir)
        correction_file = output_path / "manual_correction.csv"
        
        if not correction_file.exists():
            print(f"Manual correction file not found: {correction_file}")
            return []
        
        print("Applying manual corrections...")
        df = pd.read_csv(correction_file)
        
        # Group by image
        corrected_results = []
        failed_files = []
        
        for image_name, group in df.groupby('image_name'):
            image_path = group.iloc[0]['image_path']
            image_stem = Path(image_path).stem
            result_file = output_path / f"{image_stem}_result.json"
            
            print(f"Processing corrections for: {image_name}")
            
            if result_file.exists():
                try:
                    # Try to load the JSON file with error handling
                    with open(result_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            print(f"Warning: Empty JSON file {result_file}")
                            failed_files.append(str(result_file))
                            continue
                        
                        original_result = json.loads(content)
                    
                    # Validate required fields
                    if 'parsed_content_list' not in original_result:
                        print(f"Warning: No parsed_content_list in {result_file}")
                        failed_files.append(str(result_file))
                        continue
                    
                    if 'image_size' not in original_result:
                        print(f"Warning: No image_size in {result_file}, using default")
                        original_result['image_size'] = [1920, 1080]  # Default size
                    
                    # Apply corrections
                    corrected_parsed_content = []
                    for _, row in group.iterrows():
                        if pd.isna(row['keep_element']) or row['keep_element']:  # Handle NaN values as True
                            try:
                                element_id = int(row['element_id'])
                                if element_id < len(original_result['parsed_content_list']):
                                    original_element = original_result['parsed_content_list'][element_id]
                                    
                                    # Update content with corrected version
                                    corrected_element = original_element.copy()
                                    corrected_content = row['corrected_content']
                                    if pd.notna(corrected_content):  # Check if not NaN
                                        corrected_element['content'] = str(corrected_content)
                                    corrected_parsed_content.append(corrected_element)
                                else:
                                    print(f"Warning: Element ID {element_id} out of range for {image_name}")
                            except (ValueError, KeyError) as e:
                                print(f"Warning: Error processing element for {image_name}: {e}")
                                continue
                    
                    # Create corrected result
                    corrected_result = {
                        'image_path': image_path,
                        'image_size': tuple(original_result['image_size']) if isinstance(original_result['image_size'], list) else original_result['image_size'],
                        'parsed_content_list': corrected_parsed_content
                    }
                    corrected_results.append(corrected_result)
                    print(f"Successfully processed {image_name}: {len(corrected_parsed_content)} elements")
                    
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in {result_file}: {e}")
                    print(f"JSON error at line {e.lineno}, column {e.colno}")
                    failed_files.append(str(result_file))
                    
                    # Try to repair the JSON file by truncating at the error point
                    try:
                        print(f"Attempting to repair {result_file}...")
                        with open(result_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Try to find a valid JSON by removing lines after the error
                        for attempt in range(max(0, e.lineno - 10), len(lines)):
                            try:
                                truncated_content = ''.join(lines[:attempt]) + '}'
                                test_json = json.loads(truncated_content)
                                print(f"Successfully repaired JSON at line {attempt}")
                                
                                # Save the repaired file
                                backup_file = result_file.with_suffix('.json.backup')
                                shutil.copy2(result_file, backup_file)
                                with open(result_file, 'w', encoding='utf-8') as f:
                                    json.dump(test_json, f, indent=2)
                                print(f"Repaired file saved, backup created: {backup_file}")
                                break
                            except:
                                continue
                    except Exception as repair_error:
                        print(f"Failed to repair {result_file}: {repair_error}")
                        
                except Exception as e:
                    print(f"Error: Failed to process {result_file}: {e}")
                    failed_files.append(str(result_file))
            else:
                print(f"Warning: Result file not found: {result_file}")
                failed_files.append(str(result_file))
        
        print(f"Applied corrections to {len(corrected_results)} images")
        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for f in failed_files:
                print(f"  - {f}")
        
        return corrected_results

    def repair_json_files(self, output_dir: str):
        """Repair corrupted JSON files in the output directory"""
        output_path = Path(output_dir)
        json_files = list(output_path.glob("*_result.json"))
        
        print(f"Found {len(json_files)} JSON files to check...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"✓ {json_file.name} is valid")
            except json.JSONDecodeError as e:
                print(f"✗ {json_file.name} is corrupted at line {e.lineno}, column {e.colno}")
                
                # Try to repair
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple repair strategies
                    # 1. Try to find the last complete object before the error
                    lines = content.split('\n')
                    for i in range(len(lines) - 1, -1, -1):
                        try:
                            test_content = '\n'.join(lines[:i])
                            if test_content.strip().endswith(','):
                                test_content = test_content.strip()[:-1]  # Remove trailing comma
                            if not test_content.strip().endswith('}'):
                                test_content += '\n}'
                            
                            test_json = json.loads(test_content)
                            
                            # Backup original
                            backup_file = json_file.with_suffix('.json.corrupted')
                            shutil.copy2(json_file, backup_file)
                            
                            # Save repaired version
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(test_json, f, indent=2)
                            
                            print(f"  → Repaired {json_file.name} (backup: {backup_file.name})")
                            break
                        except:
                            continue
                    else:
                        print(f"  → Could not repair {json_file.name}")
                        
                except Exception as repair_error:
                    print(f"  → Repair failed for {json_file.name}: {repair_error}")
            except Exception as e:
                print(f"✗ {json_file.name} has other error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Collect training data for OmniParser')
    parser.add_argument('--input_dir', 
                       help='Directory containing input images (required for normal processing)')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save training data')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--box_threshold', type=float, default=0.05,
                       help='Box detection threshold')
    parser.add_argument('--manual_correction', action='store_true',
                       help='Create manual correction interface')
    parser.add_argument('--apply_corrections', action='store_true',
                       help='Apply manual corrections and generate training data')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.apply_corrections:
        # Apply corrections mode - only needs output_dir
        if not os.path.exists(args.output_dir):
            print(f"Error: Output directory {args.output_dir} does not exist")
            return
        
        print("Applying manual corrections mode")
        collector = TrainingDataCollector(device=args.device)
        corrected_results = collector.apply_manual_corrections(args.output_dir)
        if corrected_results:
            print("Generating training data from corrected annotations...")
            yolo_config, florence_data = prepare_training_data_from_omniparser_output(
                args.output_dir, corrected_results
            )
            print(f"Training data generated:")
            print(f"  YOLO config: {yolo_config}")
            print(f"  Florence2 data: {len(florence_data)} samples")
        else:
            print("No corrected data found. Make sure manual_correction.csv exists in the output directory.")
    
    else:
        # Normal processing mode - needs input_dir
        if not args.input_dir:
            print("Error: --input_dir is required for normal processing mode")
            print("Usage examples:")
            print("  Normal processing: python collect_training_data.py --input_dir ./images --output_dir ./data")
            print("  Apply corrections: python collect_training_data.py --output_dir ./data --apply_corrections")
            return
        
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            return
        
        print("Normal processing mode")
        collector = TrainingDataCollector(device=args.device)
        
        # Process images and collect raw data
        results = collector.process_image_directory(
            args.input_dir, 
            args.output_dir,
            manual_correction=args.manual_correction,
            box_threshold=args.box_threshold
        )
        
        if not args.manual_correction and results:
            # Directly generate training data without manual correction
            print("Generating training data...")
            yolo_config, florence_data = prepare_training_data_from_omniparser_output(
                args.output_dir, results
            )
            print(f"Training data generated:")
            print(f"  YOLO config: {yolo_config}")
            print(f"  Florence2 data: {len(florence_data)} samples")
            
            print(f"\nTo train the models, run:")
            print(f"python finetune_omniparser_models.py --mode both --data_dir {args.output_dir}")
        elif args.manual_correction:
            print(f"\nNext steps:")
            print(f"1. Edit the file: {args.output_dir}/manual_correction.csv")
            print(f"2. Run: python collect_training_data.py --output_dir {args.output_dir} --apply_corrections")

if __name__ == "__main__":
    main() 