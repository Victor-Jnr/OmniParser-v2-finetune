## OmniParser Fine-tuning (LoRA) — Offline Model Fix + Runbook

This guide captures the exact fixes applied and the commands to re-run the finetuning program reliably in an offline-first setup.

### What was fixed
- Replaced previously downloaded model folders with fresh weights from `microsoft/OmniParser-v2.0`.
- Enabled fully offline loading for Florence2 by placing the local modeling files alongside the weights and updating `config.json` auto-mapping:
  - `AutoConfig`: `configuration_florence2.Florence2Config`
  - `AutoModelForCausalLM`: `modeling_florence2.Florence2ForConditionalGeneration`

### One-time prerequisites
- Ensure `python` is available (if missing on your image):
```bash
update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```
- Ensure the CLI is available:
```bash
python -m pip install -U huggingface_hub  # provides huggingface-cli / hf
```

### 1) Clean and download the official weights
This follows the README’s instructions for V2 weights and renames the caption weights folder to `icon_caption_florence`.
```bash
rm -rf /workspace/weights/icon_caption_florence /workspace/weights/icon_caption /workspace/weights/icon_detect

# download the model checkpoints to local directory OmniParser/weights/
cd /workspace
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do \
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir /workspace/weights; \
done
mv -f /workspace/weights/icon_caption /workspace/weights/icon_caption_florence
```

### 2) Add Florence2 local modeling files for offline load
Florence2 expects `configuration_florence2.py` and `modeling_florence2.py` next to the weights.
```bash
cd /workspace/weights/icon_caption_florence
huggingface-cli download microsoft/Florence-2-base-ft configuration_florence2.py --local-dir . --repo-type model
huggingface-cli download microsoft/Florence-2-base-ft modeling_florence2.py --local-dir . --repo-type model
ls -la configuration_florence2.py modeling_florence2.py
```

### 3) Update auto-map entries in `config.json`
If you re-download again in the future, patch the auto-map fields to point to local files.
```bash
cd /workspace/weights/icon_caption_florence
sed -i 's#microsoft/Florence-2-base-ft--configuration_florence2.Florence2Config#configuration_florence2.Florence2Config#' config.json
sed -i 's#microsoft/Florence-2-base-ft--modeling_florence2.Florence2ForConditionalGeneration#modeling_florence2.Florence2ForConditionalGeneration#' config.json
```

### 4) Run finetuning (LoRA)
```bash
python -u /workspace/finetune_omniparser_lora.py \
  --data /workspace/training_data/florence_format/florence_data.json \
  --model_path /workspace/weights/icon_caption_florence \
  --epochs 25 \
  --batch_size 8 \
  --lr 5e-5
```

Run with log capture:
```bash
mkdir -p /workspace/logs
python -u /workspace/finetune_omniparser_lora.py \
  --data /workspace/training_data/florence_format/florence_data.json \
  --model_path /workspace/weights/icon_caption_florence \
  --epochs 25 --batch_size 8 --lr 5e-5 \
  2>&1 | tee /workspace/logs/finetune_lora_$(date +%Y%m%d_%H%M%S).log
```

Monitor:
```bash
tail -f /workspace/logs/finetune_lora_*.log | cat
```

Stop training:
```bash
pkill -f finetune_omniparser_lora.py
```

### 5) Optional: Merge the LoRA adapter into a full model
```bash
python /workspace/finetune_omniparser_lora.py --merge_only \
  --model_path /workspace/weights/icon_caption_florence \
  --lora_path /workspace/weights/icon_caption_florence_lora_finetuned \
  --merge_path /workspace/weights/icon_caption_florence_merged
```

### Outputs
- LoRA adapter: `/workspace/weights/icon_caption_florence_lora_finetuned/` (use with base model path).
- Optional merged model: `/workspace/weights/icon_caption_florence_merged/`.

### Notes
- “Warning: Using default image for item …” means some entries in `training_data/florence_format/florence_data.json` reference missing images. You can regenerate/fix training data with the included data collection and augmentation scripts.


