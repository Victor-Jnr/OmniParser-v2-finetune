# Data Augmentation for Florence2 Fine-tuning

This project now includes data augmentation capabilities to improve Florence2 model training performance by expanding the training dataset.

## Overview

The data augmentation system applies various image processing techniques to create additional training samples from your existing data, helping to improve model generalization and reduce overfitting.

## Features

### Augmentation Techniques

The system randomly applies the following image transformations:

1. **Rotation**: Random rotation within ±15 degrees
2. **Cropping**: Random crop with 0.5-3% margin (preserves content integrity)
3. **Brightness**: Random brightness adjustment (±20 pixel values)
4. **Contrast**: Random contrast scaling (0.8x to 1.2x)
5. **Gaussian Noise**: Addition of subtle gaussian noise (strength: 10)

Each augmented image receives 1-2 random transformations to create diverse variations.

## Usage

### Method 1: Integrated with Fine-tuning Script

Use the `--data_impr` parameter with the fine-tuning script:

```bash
# Run fine-tuning with 3x data augmentation
python finetune_omniparser_models_fixed.py --data_impr 3

# Run fine-tuning with 5x data augmentation  
python finetune_omniparser_models_fixed.py --data_impr 5

# Run fine-tuning without augmentation (default)
python finetune_omniparser_models_fixed.py --data_impr 1
```

### Method 2: Standalone Data Augmentation

Run data augmentation separately:

```bash
# Basic augmentation with 3x multiplier (equal weights)
python data_augmentation.py --data_dir training_data/florence_format --multiplier 3

# Weighted augmentation - favor noise and brightness
python data_augmentation.py --multiplier 3 --weight_noise 2 --weight_brightness 1

# Heavy weighting example - 8x noise, 4x crop, 2x brightness
python data_augmentation.py --multiplier 5 --weight_noise 3 --weight_crop 2 --weight_brightness 1

# Custom directory with weights
python data_augmentation.py --data_dir /path/to/your/data --multiplier 5 --weight_contrast 2
```

#### Weight System Explanation

Each augmentation type has a weight (0-10):
- **Weight 0**: Default probability (1 entry in selection pool)
- **Weight 1**: Double probability (2 entries)  
- **Weight 2**: Quadruple probability (4 entries)
- **Weight 3**: 8x probability (8 entries)
- **Weight N**: 2^N probability

Available augmentation types:
- `--weight_crop`: Random cropping (0.5-3% margin)
- `--weight_brightness`: Brightness adjustment (±20 pixels)
- `--weight_contrast`: Contrast scaling (0.8x-1.2x)
- `--weight_noise`: Gaussian noise (strength: 10)
- `--weight_scaling`: Random scaling (-20% to +5%)

### Method 3: Data Cleanup

After augmentation, you may want to manually review and delete unwanted images from the `imgs/` directory. Use the cleanup option to automatically remove corresponding JSON entries:

```bash
# Clean up JSON data after manually deleting unwanted images
python data_augmentation.py --clean_missing

# Clean up with custom directory
python data_augmentation.py --data_dir /path/to/your/data --clean_missing
```

**Cleanup Workflow:**
1. Run data augmentation: `python data_augmentation.py --multiplier 3`
2. Review generated images in `training_data/florence_format/imgs/`
3. Manually delete any unwanted augmented images
4. Run cleanup: `python data_augmentation.py --clean_missing`
5. Proceed with training using the cleaned dataset

## Data Format Requirements

Your training data should follow this format in `florence_data.json`:

```json
[
    {
        "image_path": "raw_images/example.png",
        "content": "button icon",
        "bbox": [0.1, 0.2, 0.3, 0.4]
    }
]
```

- `image_path`: Path to the original image
- `content`: Text description/label for the UI element
- `bbox`: Optional bounding box coordinates [x1, y1, x2, y2] in normalized format

## Output Structure

After augmentation:

```
training_data/florence_format/
├── florence_data.json              # Updated with augmented entries
├── florence_data_original.json     # Backup of original data
└── imgs/                           # Augmented images
    ├── example_aug_1_rot_5.2.png
    ├── example_aug_1_bright_-10_noise_10.png
    └── ...
```

## Example Output

Original data: 20 samples → 3x augmentation = 60 total samples

```
Original data: 93 items
Augmented data: 279 items  
Multiplier achieved: 3.00x
```

## Technical Details

### Processing Logic
1. **Crop First**: Each bbox region is cropped from the original image
2. **Augment Cropped Region**: Apply transformations only to the cropped UI element
3. **Update Coordinates**: Set bbox to [0, 0, 1, 1] since the cropped image now contains the full element
4. **Resize to Standard**: All cropped images are resized to 64x64 pixels (following OmniParser standard)

### Why This Approach Works
- **Coordinate Accuracy**: No matter how we rotate/transform the cropped image, the coordinates always point to the correct content
- **Content Preservation**: UI elements remain fully visible regardless of transformation
- **Training Efficiency**: Models train on focused UI elements rather than full screenshots

### Safety Features
- Automatic backup of original training data
- Crop limits prevent content loss (max 3% margin)
- Rotation limits preserve text readability (±15°)
- Brightness/contrast ranges maintain visibility

### Performance Considerations
- Processes images sequentially to avoid memory issues
- Progress indicators for large datasets
- Error handling for corrupted/missing images

## Recommended Settings

| Dataset Size | Recommended Multiplier | Expected Training Time |
|--------------|----------------------|----------------------|
| < 50 samples | 5-10x | +200-400% |
| 50-200 samples | 3-5x | +100-200% |
| > 200 samples | 2-3x | +50-100% |

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or multiplier
2. **Image not found**: Check image paths are relative to project root
3. **Poor augmentation quality**: Adjust parameters in `data_augmentation.py`
4. **Inconsistent data after manual deletion**: Use `--clean_missing` to fix JSON data

### Quality Verification

Check a few augmented images manually:
```bash
ls training_data/florence_format/imgs/
```

The filename format `original_content_aug_N_technique_params.png` shows what transformations were applied.

### Data Cleanup Features

After generating augmented images, you can:

1. **Manual Review**: Examine each generated image for quality
2. **Selective Deletion**: Delete unwanted images directly from the filesystem
3. **Automatic Cleanup**: Use `--clean_missing` to remove orphaned JSON entries

**Example Cleanup Session:**
```bash
# Generate augmented data
python data_augmentation.py --multiplier 5

# Review and manually delete poor quality images
ls training_data/florence_format/imgs/
rm training_data/florence_format/imgs/poor_quality_image.png

# Clean up JSON data
python data_augmentation.py --clean_missing

# Verify final data count
python -c "import json; data=json.load(open('training_data/florence_format/florence_data.json')); print(f'Final entries: {len(data)}')"
```

## Benefits

- **Improved Generalization**: Model learns from varied visual conditions
- **Reduced Overfitting**: More diverse training samples
- **Better Robustness**: Handles lighting, rotation, and noise variations
- **Higher Accuracy**: Especially beneficial for small datasets

## Integration with OmniParser

The augmented data works seamlessly with the existing OmniParser training pipeline:
- Maintains Florence2 format compatibility
- Preserves bounding box relationships
- Uses standard 64x64 image dimensions
- Compatible with existing evaluation scripts