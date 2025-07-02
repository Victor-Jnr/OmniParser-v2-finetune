#!/usr/bin/env python3
"""
JSON File Repair Tool for OmniParser Training Data

This script repairs corrupted JSON files in the training data directory.
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def repair_json_file(json_file_path):
    """Attempt to repair a corrupted JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as-is first
        try:
            json.loads(content)
            print(f"✓ {json_file_path.name} is already valid")
            return True
        except json.JSONDecodeError as e:
            print(f"✗ {json_file_path.name} is corrupted at line {e.lineno}, column {e.colno}")
        
        # Repair strategies
        lines = content.split('\n')
        
        # Strategy 1: Find the last valid JSON by removing lines from the end
        for i in range(len(lines) - 1, -1, -1):
            try:
                test_content = '\n'.join(lines[:i])
                
                # Clean up common issues
                test_content = test_content.strip()
                if test_content.endswith(','):
                    test_content = test_content[:-1]  # Remove trailing comma
                
                # Ensure proper closure
                if not test_content.endswith('}'):
                    test_content += '\n}'
                
                test_json = json.loads(test_content)
                
                # Backup original
                backup_file = json_file_path.with_suffix('.json.corrupted')
                shutil.copy2(json_file_path, backup_file)
                
                # Save repaired version
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(test_json, f, indent=2)
                
                print(f"  → Repaired {json_file_path.name} (backup: {backup_file.name})")
                return True
                
            except:
                continue
        
        # Strategy 2: Try to rebuild minimal structure
        try:
            minimal_json = {
                "image_path": "unknown",
                "image_size": [1920, 1080],
                "parsed_content_list": [],
                "label_coordinates": []
            }
            
            backup_file = json_file_path.with_suffix('.json.corrupted')
            shutil.copy2(json_file_path, backup_file)
            
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_json, f, indent=2)
            
            print(f"  → Created minimal structure for {json_file_path.name} (backup: {backup_file.name})")
            return True
            
        except Exception as e:
            print(f"  → Could not repair {json_file_path.name}: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error reading {json_file_path.name}: {e}")
        return False

def repair_directory(output_dir):
    """Repair all JSON files in the output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    json_files = list(output_path.glob("*_result.json"))
    
    if not json_files:
        print("No JSON result files found")
        return
    
    print(f"Found {len(json_files)} JSON files to check...")
    
    repaired_count = 0
    for json_file in json_files:
        if repair_json_file(json_file):
            repaired_count += 1
    
    print(f"\nRepair summary:")
    print(f"Total files: {len(json_files)}")
    print(f"Successfully processed: {repaired_count}")
    print(f"Failed: {len(json_files) - repaired_count}")

def main():
    parser = argparse.ArgumentParser(description='Repair corrupted JSON files')
    parser.add_argument('--output_dir', required=True,
                       help='Directory containing JSON files to repair')
    
    args = parser.parse_args()
    repair_directory(args.output_dir)

if __name__ == "__main__":
    main() 