#!/usr/bin/env python3
"""
OmniParser Demo Script
Converted from demo_nofile.ipynb

This script demonstrates the OmniParser functionality for processing images
and extracting UI elements with bounding boxes and captions.
"""

import os
import time
import importlib
import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

def main(image_path):
    image_path = image_path
    # Configuration
    device = 'cuda'
    model_path = 'weights/icon_detect/model.pt'
    BOX_THRESHOLD = 0.05
    
    # Initialize SOM model
    print("Loading SOM model...")
    som_model = get_yolo_model(model_path)
    som_model.to(device)
    print(f'Model loaded to {device}')
    
    # Initialize caption model (using Florence2)
    print("Loading caption model...")
    
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        # model_name_or_path="weights/icon_caption_florence", 
        model_name_or_path="weights/icon_caption_florence_finetuned", 
        device=device
    )
    print("Caption model loaded")
    
    # Image selection (you can modify this to use different images)
    image_options = [
        'imgs/google_page.png',
        'imgs/windows_home.png',
        'imgs/windows_multitab.png',
        'imgs/omni3.jpg',
        'imgs/ios.png',
        'imgs/word.png'
    ]
    
    # Use word.png as default (as shown in the notebook)
    # image_path = 'imgs/live.png'
    
    # Load and process image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    print(f'Image size: {image.size}')
    
    # Configure bounding box drawing parameters
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    
    # Perform OCR
    print("Performing OCR...")
    start_time = time.time()
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
        use_paddleocr=True
    )
    text, ocr_bbox = ocr_bbox_rslt
    ocr_time = time.time() - start_time
    print(f"OCR completed in {ocr_time:.2f} seconds")
    
    # Get SOM labeled image with captions
    print("Generating SOM labeled image...")
    start_time = time.time()
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path, 
        som_model, 
        BOX_TRESHOLD=BOX_THRESHOLD, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        use_local_semantics=True, 
        iou_threshold=0.7, 
        scale_img=False, 
        # batch_size=128
    )
    caption_time = time.time() - start_time
    print(f"Caption generation completed in {caption_time:.2f} seconds")
    
    # Save the labeled image
    output_image_path = image_path.replace('.png', '_labeled.png').replace('.jpg', '_labeled.jpg')
    
    # Handle case where dino_labeled_img might be a base64 string, file path, or PIL Image
    if isinstance(dino_labeled_img, str):
        # Check if it's a base64 encoded image (starts with common image data URI prefixes or is very long)
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
                labeled_img.save(output_image_path)
                print(f"Labeled image decoded from base64 and saved to: {output_image_path}")
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
                print(f"String starts with: {dino_labeled_img[:100]}...")
        else:
            # If it's a file path, load and save to our desired location
            try:
                labeled_img = Image.open(dino_labeled_img)
                labeled_img.save(output_image_path)
                print(f"Labeled image loaded from {dino_labeled_img} and saved to: {output_image_path}")
            except Exception as e:
                print(f"Error loading image from path: {e}")
    else:
        # If it's a PIL Image, save directly
        try:
            dino_labeled_img.save(output_image_path)
            print(f"Labeled image saved to: {output_image_path}")
        except Exception as e:
            print(f"Error saving PIL Image: {e}")
    
    # Create DataFrame from parsed content
    df = pd.DataFrame(parsed_content_list)
    df['ID'] = range(len(df))
    
    # Display results
    print("\nParsed Content Summary:")
    print(f"Total elements detected: {len(parsed_content_list)}")
    
    print("\nLabel Coordinates:")
    print(label_coordinates)
    
    print("\nDataFrame:")
    print(df.to_string())
    
    print("\nDetailed parsed content:")
    for i, content in enumerate(parsed_content_list):
        print(f"Element {i}: {content}")
    
    # Print return content structure
    print("\nReturn Content Structure:")
    print(f"1. Labeled Image: {type(dino_labeled_img)} - saved to {output_image_path}")
    print(f"2. Label Coordinates: {type(label_coordinates)} with {len(label_coordinates) if hasattr(label_coordinates, '__len__') else 'N/A'} items")
    print(f"3. Parsed Content List: {type(parsed_content_list)} with {len(parsed_content_list)} items")
    print(f"4. DataFrame: {type(df)} with shape {df.shape}")
    
    return dino_labeled_img, label_coordinates, parsed_content_list, df

if __name__ == "__main__":
    # for file in os.listdir('imgs'):
    #     if file.endswith('.png') or file.endswith('.jpg') and not file.endswith('_labeled.png') and not file.endswith('_labeled.jpg') and not file.startswith('demo') :
    #         image_path = os.path.join('imgs', file)
    #         main(image_path) 
    # main('imgs/vod_play_detail_full_screen.png')
    main('imgs/live_channel_labeled.png')