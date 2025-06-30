# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniParser is a comprehensive screen parsing tool that converts GUI screenshots into structured, interactable elements for vision-based AI agents. It uses a three-layer detection architecture combining OCR text detection, YOLO icon detection, and Florence2 semantic understanding to parse user interfaces.

## Core Architecture

### Three-Layer Processing Pipeline
1. **OCR Layer (PaddleOCR)**: Detects and extracts text regions with content
2. **Icon Detection Layer (YOLO)**: Identifies interactable UI elements/icons 
3. **Semantic Layer (Florence2)**: Generates functional descriptions for detected icons
4. **Fusion Layer**: Merges overlapping detections and assigns appropriate content sources

### Key Components
- `util/omniparser.py`: Main OmniParser class with parse() method
- `util/utils.py`: Core processing functions including `get_som_labeled_img()`, `remove_overlap_new()`, and `get_parsed_content_icon()`
- `gradio_demo.py`: Simple Gradio interface for testing parsing functionality
- `omnitool/`: Complete agent framework with Windows VM control capabilities

## Development Commands

### Environment Setup
```bash
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

### Download Model Weights
```bash
# Download V2 model checkpoints
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do 
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence
```

### Running Demos
```bash
# Basic Gradio demo
python gradio_demo.py

# OmniTool agent interface
python omnitool/gradio/app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000

# Streamlit interface  
python omnitool/gradio/app_streamlit.py
```

### Linting and Formatting  
```bash
ruff check .
ruff format .
```

## Model Training and Fine-tuning

### YOLO Model Training
- Training configurations in `yolo_training/` directory
- Custom training data format in `training_data/yolo_format/`
- Use `finetune_omniparser_models_fixed.py` for model fine-tuning

### Florence2 Model Training
- Training data format in `training_data/florence_format/`
- Manual corrections tracked in `training_data/manual_correction.csv`

## Key File Locations

### Core Processing
- `util/utils.py`: Main processing functions (`get_som_labeled_img`, `remove_overlap_new`, `check_ocr_box`)
- `util/omniparser.py`: OmniParser class implementation
- `weights/`: Model checkpoints (icon_detect/, icon_caption_florence/)

### Agent Framework (OmniTool)
- `omnitool/gradio/app.py`: Main agent interface
- `omnitool/gradio/agent/`: Agent implementations (anthropic_agent.py, vlm_agent.py)
- `omnitool/gradio/tools/`: Tool implementations (computer.py, screen_capture.py)
- `omnitool/omnibox/`: Windows VM container setup

### Training and Evaluation
- `training_data/`: Training datasets in YOLO and Florence formats
- `eval/`: Evaluation scripts and benchmark results
- `yolo_training/`: YOLO training runs and results

## Architecture Notes

### Element Source Types
- `box_ocr_content_ocr`: Pure OCR text regions (not interactable)
- `box_yolo_content_ocr`: YOLO-detected icons containing OCR text (interactable)  
- `box_yolo_content_yolo`: YOLO-detected icons with Florence2-generated descriptions (interactable)

### Processing Flow
```
Input Image → PaddleOCR Text Detection → YOLO Icon Detection → 
Overlap Processing & Fusion → Florence2 Semantic Generation → Structured Output
```

### Key Configuration Parameters
- `BOX_TRESHOLD`: YOLO detection confidence threshold (default: varies by model)
- `iou_threshold`: Intersection-over-Union threshold for overlap removal (default: 0.7)
- `batch_size`: Batch size for Florence2 processing (default: 128)
- `use_local_semantics`: Enable Florence2 semantic generation (default: True)

## Development Workflow

1. **Core parsing development**: Work with `util/` modules and test with `gradio_demo.py`
2. **Agent development**: Use `omnitool/gradio/` components  
3. **Model training**: Use training scripts and data in `training_data/` and `yolo_training/`
4. **Evaluation**: Run scripts in `eval/` directory

## Important Implementation Details

- Models expect specific input formats: YOLO uses normalized coordinates, Florence2 expects 64x64 cropped images
- Overlap processing is critical - OCR text inside YOLO boxes gets merged as `box_yolo_content_ocr` 
- Florence2 generates semantic descriptions only for YOLO boxes without OCR text
- The system supports both GPU and CPU inference with automatic device selection