## Training Data Format (YOLO + Florence2)

This document consolidates all required formats for OmniParser training data.

### Overview
- Two datasets are used:
  - **YOLO (icon detection)**: images + normalized bounding boxes for a single class (icon)
  - **Florence2 (icon captioning)**: JSON list of items with `image_path`, textual `content`, and optional `bbox`

---

### Common requirements (images)
- File types: `.png`, `.jpg`, `.jpeg`
- Color: RGB (loader converts as needed)
- Size: any; for Florence2 the ROI is resized to 64×64 during training
- Paths: `image_path` must exist (absolute or project-root relative). Missing files trigger “Using default image …” warnings
- Coordinates are normalized floats in [0, 1]

---

### YOLO detection format
Directory layout (as used in this project):
```
training_data/
└── yolo_format/
    ├── train/
    │   ├── image1.jpg
    │   ├── image1.txt   # each line: class_id cx cy w h (normalized)
    │   └── ...
    ├── val/
    │   ├── image2.jpg
    │   ├── image2.txt
    │   └── ...
    └── dataset.yaml
```

Label file (`.txt`) example for `image1.txt`:
```
0 0.500 0.300 0.100 0.100
0 0.700 0.800 0.050 0.050
```
- Format per line: `class_id center_x center_y width height`
- Normalization: all values in [0, 1] relative to image width/height
- Class mapping: single class `0` → icon

Minimal `dataset.yaml` example:
```yaml
path: ./training_data/yolo_format
train: train
val: val
nc: 1
names: [icon]
```

---

### Florence2 caption format
Top-level file: `training_data/florence_format/florence_data.json`

Schema (array of objects):
```json
[
  {
    "image_path": "path/to/image.jpg",
    "content": "Settings button",
    "bbox": [0.45, 0.25, 0.55, 0.35]
  }
]
```
- **image_path** (string, required): path to the source screenshot or cropped icon image
- **content** (string, required): short natural-language label/description of the UI icon (e.g., "search icon", "close button")
- **bbox** (array[4], optional): `[x1, y1, x2, y2]` normalized to [0, 1]
  - If present, the ROI is cropped and resized to 64×64 before training
  - If absent, the full image is used and resized to 64×64

Notes:
- Items with missing `image_path` will be skipped at runtime (you'll see “Using default image …” warnings)
- Ensure `x2 > x1` and `y2 > y1` and all values are within [0, 1]

---

### Data augmentation behavior (Florence2)
- ROI is cropped using `bbox`, augmented (rotation, crop margin, brightness, contrast, gaussian noise, scaling), then resized to 64×64
- Augmented images are saved under `training_data/florence_format/imgs/`
- Coordinates for augmented crops are set to `[0, 0, 1, 1]` (full image)
- See `DATA_AUGMENTATION_README.md` for options and workflow

---

### Quality & validation checklist
- Image exists at `image_path` and can be opened
- `content` is concise and relevant to the icon’s function
- `bbox` (if provided) is normalized `[x1,y1,x2,y2]` with valid ordering and range
- YOLO labels have normalized `cx,cy,w,h` in [0, 1]

Quick validation (Python):
```python
import json, os

data = json.load(open('training_data/florence_format/florence_data.json', 'r', encoding='utf-8'))
bad = []
for i, item in enumerate(data):
    p = item.get('image_path')
    bbox = item.get('bbox')
    ok = True
    if not p or not os.path.exists(p):
        ok = False
    if bbox is not None:
        ok = ok and isinstance(bbox, list) and len(bbox) == 4 and \
             all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bbox) and \
             bbox[2] > bbox[0] and bbox[3] > bbox[1]
    if not ok:
        bad.append((i, item))
print(f'Invalid entries: {len(bad)}')
```

If you deleted bad images after augmentation, run:
```bash
python data_augmentation.py --clean_missing
```

---

### Practical examples
Florence2 JSON item examples:
```json
{ "image_path": "training_data/florence_format/imgs/search.png", "content": "search icon", "bbox": [0.62,0.08,0.69,0.15] }
{ "image_path": "training_data/florence_format/imgs/close.png",   "content": "close button" }
```

YOLO label examples (single-class):
```
0 0.523 0.114 0.056 0.042
0 0.312 0.774 0.081 0.066
```

---

### Troubleshooting
- “Using default image for item …”: image file missing → fix path or run cleanup
- YOLO not training: verify `dataset.yaml` paths and label normalization
- Florence2 poor results: ensure `content` quality and correct `bbox` regions; consider augmentation