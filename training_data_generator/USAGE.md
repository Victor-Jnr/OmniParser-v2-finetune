## Training Data Generator (GPT-4o) + Fine-tune Runner

This folder contains a small pipeline to:
- Label your screenshots using GPT-4o (vision) with normalized bboxes and short icon labels.
- Write Florence2-compatible training data to an isolated directory (under this folder).
- Run Florence2 LoRA fine-tuning on the generated dataset.

### Output locations (isolated)
- Florence2 JSON is written to: `/workspace/training_data_generator/training_data/florence_format/florence_data.json`
- Keep your raw screenshots anywhere. Two common options:
  - `/workspace/raw_images/`
  - `/workspace/training_data_generator/raw_images/` (works fine too; see commands below)

### Environment
Set the following env vars (you can mirror `sample.env` to your `.env`):
```
GPT_MODEL=gpt-4o:2024-11-20
GPT_API_KEY=sk-...
```
Export before running (choose one file):
```bash
export $(cat /workspace/training_data_generator/.env | xargs)
# or
export $(cat /workspace/training_data_generator/sample.env | xargs)
```

### Install Python requirements for the generator/trainer
Install once:
```bash
python -m pip install -r /workspace/training_data_generator/trainer_req.txt
```

### 1) Generate training data via GPT labels (keeps bboxes)

Using raw images under `/workspace/raw_images/`:
```bash
python /workspace/training_data_generator/gpt_labeler.py \
  --raw_dir /workspace/raw_images \
  --out_json /workspace/training_data_generator/training_data/florence_format/florence_data.json
```

Using raw images under `/workspace/training_data_generator/raw_images/`:
```bash
python /workspace/training_data_generator/gpt_labeler.py \
  --raw_dir /workspace/training_data_generator/raw_images \
  --out_json /workspace/training_data_generator/training_data/florence_format/florence_data.json
```

If you have subfolders in your raw images directory, add `--recursive`.

Behavior:
- Sends each screenshot (base64) to GPT-4o with instructions to return `{ items: [{ label, bbox }] }`.
- Validates and clamps bbox to [0,1]; ensures `x2>x1, y2>y1`.
- Appends entries to the output JSON in Florence2 format:
  ```json
  { "image_path": "/path/to/screenshot.png", "content": "terminal icon", "bbox": [x1,y1,x2,y2] }
  ```

### 2) Fine-tune Florence2 (LoRA) on generated data
```bash
python /workspace/training_data_generator/run_finetune.py \
  --data /workspace/training_data_generator/training_data/florence_format/florence_data.json \
  --model_path /workspace/weights/icon_caption_florence \
  --epochs 25 --batch_size 8 --lr 5e-5
```

Outputs:
- LoRA adapter written to `/workspace/weights/icon_caption_florence_lora_finetuned/`.

### 3) (Optional) Merge LoRA into a standalone model
```bash
python /workspace/finetune_omniparser_lora.py --merge_only \
  --model_path /workspace/weights/icon_caption_florence \
  --lora_path /workspace/weights/icon_caption_florence_lora_finetuned \
  --merge_path /workspace/weights/icon_caption_florence_merged
```

### Labeling best practices
- Keep labels short and consistent (e.g., "terminal icon", "settings gear").
- Provide tight bboxes with ~1–3% margin; too tiny boxes are discouraged.
- De-duplicate highly overlapping boxes if you post-process.

### Troubleshooting
- Missing `GPT_API_KEY`: export from `sample.env` or set in your environment.
- No images found: check `--raw_dir` and file extensions.
- Fine-tune fails: ensure `/workspace/weights/icon_caption_florence` exists and is offline-ready.


