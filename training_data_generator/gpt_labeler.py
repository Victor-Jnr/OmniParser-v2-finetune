#!/usr/bin/env python3
"""
GPT-based Florence2 training data generator.

Reads images from ./raw_images/ (recursively optional), sends base64 screenshots to a GPT-4o model
to get icon labels with normalized bounding boxes, and writes/updates
./training_data_generator/training_data/florence_format/florence_data.json.

Environment:
  - GPT_MODEL (e.g., gpt-4o:2024-11-20)
  - GPT_API_KEY (OpenAI-style key)

Usage:
  python training_data_generator/gpt_labeler.py \
    --raw_dir ./raw_images \
    --out_json ./training_data_generator/training_data/florence_format/florence_data.json \
    [--recursive]

Notes:
  - Keeps bounding boxes to improve Florence2 fine-tuning quality.
  - Ensures bbox normalization/clamping and valid ordering.
"""

import os
import io
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image


PROMPT = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": (
                "You are labeling UI icons. Return JSON ONLY.\n\n"
                "Rules:\n"
                "- List distinct UI icons with short semantic labels (e.g., 'terminal icon', 'settings gear').\n"
                "- Provide normalized bounding boxes [x1,y1,x2,y2] in [0,1] with x2>x1, y2>y1.\n"
                "- Keep labels concise and consistent.\n\n"
                "Schema:\n"
                "{\n  'items': [ { 'label': 'string', 'bbox': [x1,y1,x2,y2] } ]\n}\n"
            )
        },
        # image will be appended programmatically
    ],
}


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def clamp_float(v: float) -> float:
    v = float(v)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def sanitize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clamp bbox entries from GPT output."""
    cleaned: List[Dict[str, Any]] = []
    for it in items or []:
        label = str(it.get("label", "")).strip()
        bbox = it.get("bbox")
        if not label or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([clamp_float(x1), clamp_float(x2)])
        y1, y2 = sorted([clamp_float(y1), clamp_float(y2)])
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            continue
        cleaned.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return cleaned


def load_existing(out_json: Path) -> List[Dict[str, Any]]:
    if out_json.exists():
        try:
            return json.load(out_json.open("r", encoding="utf-8"))
        except Exception:
            return []
    return []


def call_gpt_vision(model: str, api_key: str, image_b64: str) -> Dict[str, Any]:
    """Call GPT-4o (or compatible) with a text+image prompt; return parsed JSON.

    Assumes OpenAI-compatible HTTP API via openai>=1.x.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Please `pip install openai>=1.0.0`.") from e

    client = OpenAI(api_key=api_key)

    # Build message with image content part
    content = list(PROMPT["content"])  # shallow copy
    content.append({
        "type": "input_image",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
        }
    })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=600,
    )

    text = resp.choices[0].message.content or "{}"
    # try to locate JSON in response
    text = text.strip()
    # Many providers may wrap in markdown fences; try to extract
    if text.startswith("```"):
        # remove first and last fence
        lines = text.splitlines()
        # drop leading fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop trailing fence
        while lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        # fallback to empty
        return {"items": []}


def main():
    parser = argparse.ArgumentParser(description="Generate Florence2 training data using GPT vision")
    parser.add_argument("--raw_dir", default="./raw_images", help="Directory with raw screenshots")
    parser.add_argument("--out_json", default="./training_data_generator/training_data/florence_format/florence_data.json",
                        help="Output Florence2 JSON path")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories for images")
    args = parser.parse_args()

    model = os.getenv("GPT_MODEL", "gpt-4o:2024-11-20")
    api_key = os.getenv("GPT_API_KEY", "")
    if not api_key:
        print("Error: GPT_API_KEY not set in environment.")
        print("Set it or put it in training_data_generator/sample.env and export before running.")
        sys.exit(1)

    raw_dir = Path(args.raw_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Error: raw_dir not found: {raw_dir}")
        sys.exit(1)

    # Collect images
    images: List[Path] = []
    if args.recursive:
        for p in raw_dir.rglob("*"):
            if p.is_file() and is_image_file(p):
                images.append(p)
    else:
        for p in raw_dir.glob("*"):
            if p.is_file() and is_image_file(p):
                images.append(p)

    if not images:
        print(f"No images found in {raw_dir}")
        sys.exit(0)

    existing = load_existing(out_json)
    added = 0

    for img_path in images:
        try:
            # load once to ensure readable and to confirm dimensions
            Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skip unreadable image: {img_path} ({e})")
            continue

        img_b64 = encode_image_to_base64(str(img_path))
        result = call_gpt_vision(model=model, api_key=api_key, image_b64=img_b64)
        items = sanitize_items(result.get("items", []))
        if not items:
            print(f"No items for {img_path.name}")
            continue

        for it in items:
            existing.append({
                "image_path": str(img_path),
                "content": it["label"],
                "bbox": it["bbox"],
            })
            added += 1

        # Persist progressively to avoid data loss
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        print(f"Labeled {img_path.name}: +{len(items)} entries (total so far: {len(existing)})")

    print(f"\nDone. Added {added} entries. Output: {out_json}")


if __name__ == "__main__":
    main()


