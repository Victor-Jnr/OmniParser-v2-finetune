#!/usr/bin/env python3
"""
POCO UI Coordinate Tagging Script

This script tags UI elements on screenshots based on POCO UI identification results.
It visualizes normalized coordinates by converting them to actual pixel positions.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import json

def parse_poco_data(poco_data):
    """Parse POCO UI data and extract coordinate information"""
    return {
        'type': poco_data.get('type', ''),
        'name': poco_data.get('name', ''),
        'text': poco_data.get('text', ''),
        'pos': poco_data.get('pos', [0, 0]),  # Center position (normalized)
        'boundsInParent': poco_data.get('boundsInParent', [0, 0]),  # Top-left corner (normalized)
        'anchorPoint': poco_data.get('anchorPoint', [0.5, 0.5]),  # Anchor point within element
        'size': poco_data.get('size', [0, 0]),  # Width and height (normalized)
        'enabled': poco_data.get('enabled', False),
        'visible': poco_data.get('visible', False),
        'selected': poco_data.get('selected', False)
    }

def normalize_to_pixel(normalized_coords, image_width, image_height):
    """Convert normalized coordinates to pixel coordinates"""
    if len(normalized_coords) == 2:
        x_norm, y_norm = normalized_coords
        return [int(x_norm * image_width), int(y_norm * image_height)]
    return normalized_coords

def draw_ui_element(image, element_data, image_width, image_height):
    """Draw UI element boundaries and information on the image"""
    # Convert normalized coordinates to pixels
    pos_px = normalize_to_pixel(element_data['pos'], image_width, image_height)
    bounds_px = normalize_to_pixel(element_data['boundsInParent'], image_width, image_height)
    size_px = normalize_to_pixel(element_data['size'], image_width, image_height)
    
    # Calculate bounding box
    x1, y1 = bounds_px
    width, height = size_px
    x2, y2 = x1 + width, y1 + height
    
    # Draw element rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    
    # Draw center point
    cv2.circle(image, tuple(pos_px), 5, (255, 0, 0), -1)  # Blue center dot
    
    # Draw anchor point (relative to element bounds)
    anchor_x = x1 + int(element_data['anchorPoint'][0] * width)
    anchor_y = y1 + int(element_data['anchorPoint'][1] * height)
    cv2.circle(image, (anchor_x, anchor_y), 3, (0, 0, 255), -1)  # Red anchor dot
    
    return image, (x1, y1, x2, y2, pos_px, (anchor_x, anchor_y))

def add_text_annotations(image, element_data, coords_info):
    """Add text annotations with coordinate information"""
    x1, y1, x2, y2, pos_px, anchor_px = coords_info
    
    # Convert to PIL for better text rendering
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        # Try to use a better font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Main element info
    text_lines = [
        f"Text: {element_data['text']}",
        f"Type: {element_data['type'].split('.')[-1]}",
        f"Pos: {element_data['pos']} → {pos_px}px",
        f"Bounds: {element_data['boundsInParent']} → ({x1},{y1})px",
        f"Size: {element_data['size']} → {x2-x1}×{y2-y1}px",
        f"Anchor: {element_data['anchorPoint']} → ({anchor_px[0]},{anchor_px[1]})px"
    ]
    
    # Draw background for text
    text_y = max(10, y1 - 120)
    text_x = max(10, x1)
    
    # Calculate text background size
    max_width = max([draw.textlength(line, font=small_font) for line in text_lines])
    text_bg_height = len(text_lines) * 15 + 10
    
    # Draw text background
    draw.rectangle([text_x-5, text_y-5, text_x + max_width + 10, text_y + text_bg_height], 
                   fill=(0, 0, 0, 180))
    
    # Draw text lines
    for i, line in enumerate(text_lines):
        draw.text((text_x, text_y + i * 15), line, fill=(255, 255, 255), font=small_font)
    
    # Add coordinate labels
    draw.text((pos_px[0] + 8, pos_px[1] - 8), "CENTER", fill=(255, 255, 0), font=small_font)
    draw.text((anchor_px[0] + 8, anchor_px[1] - 8), "ANCHOR", fill=(255, 0, 0), font=small_font)
    draw.text((x1, y1 - 20), "TOP-LEFT", fill=(0, 255, 0), font=small_font)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def tag_ui_coordinates(image_path, output_path, poco_data):
    """Main function to tag UI coordinates on image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}×{height}")
    
    # Parse POCO data
    element_data = parse_poco_data(poco_data)
    
    # Draw UI element
    image, coords_info = draw_ui_element(image, element_data, width, height)
    
    # Add text annotations
    image = add_text_annotations(image, element_data, coords_info)
    
    # Save result
    cv2.imwrite(output_path, image)
    print(f"Tagged image saved to: {output_path}")
    
    return image

def main():
    # POCO UI data from your example
    poco_data = {
        'type': 'android.widget.TextView',
        'name': 'com.integration.unitvsiptv:id/mTextTitle',
        'text': 'FEATURED',
        'enabled': True,
        'visible': True,
        'checkable': False,
        'pos': [0.2078125, 0.17916666666666667],
        'resourceId': b'com.integration.unitvsiptv:id/mTextTitle',
        'scrollable': False,
        'boundsInParent': [0.1015625, 0.0625],
        'selected': True,
        'anchorPoint': [0.5, 0.5],
        'size': [0.1015625, 0.05694444444444444],
        'zOrders': {'global': 0, 'local': 1},
        'editalbe': False,
        'checked': False,
        'focused': False,
        'touchable': False,
        'package': b'com.integration.unitvsiptv',
        'scale': [1, 1],
        'dismissable': False,
        'longClickable': False,
        'focusable': False
    }
    
    # Process the image
    input_path = "imgs/UVFreeFeatre.png"
    output_path = "imgs/UVFreeFeatre_tagged.png"
    
    try:
        tagged_image = tag_ui_coordinates(input_path, output_path, poco_data)
        print("✅ Successfully tagged UI coordinates!")
        print("\nCoordinate Explanations:")
        print("🔹 pos: Normalized center position of the element")
        print("🔹 boundsInParent: Normalized top-left corner position")
        print("🔹 anchorPoint: Relative anchor point within the element (0.5,0.5 = center)")
        print("🔹 size: Normalized width and height of the element")
        print("\nVisualization:")
        print("🟢 Green rectangle: Element boundaries")
        print("🔵 Blue dot: Center position (pos)")
        print("🔴 Red dot: Anchor point")
        print("🟡 Yellow text: CENTER label")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()