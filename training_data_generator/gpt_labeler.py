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
import math


# Stricter prompt (without constraining labels to a fixed vocabulary)
STRICT_PROMPT_TEXT = (
    "You are labeling ALL UI icons in the image. Return JSON ONLY.\n\n"
    "DEFINITIONS:\n"
    "- Icon: a non-text glyph/button (toolbar, tray/dock, title bar controls, sidebar, app chrome).\n"
    "- Include very small glyphs (status, wifi, battery, volume, bluetooth, time, close/min/max).\n"
    "- Exclude plain text unless it is part of an icon.\n\n"
    "REQUIREMENTS:\n"
    "- Be exhaustive across the whole screen. Do not skip tiny icons.\n"
    "- Provide tight bounding boxes with ~1–3% margin.\n"
    "- Normalized bbox: [x1,y1,x2,y2] in [0,1], x2>x1, y2>y1.\n"
    "- Minimum size: keep icons with min(width,height) ≥ 0.01 or area ≥ 0.0004.\n"
    "- For each item, include a short, consistent label (e.g., 'terminal icon', 'settings gear').\n"
    "- Also include a 'confidence' field in [0,1] and a high-level 'category' (e.g., 'system','window_control','app_toolbar','status_tray','navigation','other').\n"
    "- Avoid near-duplicate boxes; if overlapping icons exist, keep both with distinct labels.\n\n"
    "OUTPUT JSON SCHEMA ONLY:\n"
    "{\n"
    "  \"items\": [\n"
    "    { \"label\": \"string\", \"bbox\": [x1,y1,x2,y2], \"confidence\": float, \"category\": \"string\" }\n"
    "  ]\n"
    "}\n"
)

PROMPT = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": STRICT_PROMPT_TEXT
        },
        # image will be appended programmatically
    ],
}


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
        conf = it.get("confidence")
        try:
            conf = float(conf) if conf is not None else 0.5
        except Exception:
            conf = 0.5
        cleaned.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": conf})
    return cleaned


def load_existing(out_json: Path) -> List[Dict[str, Any]]:
    if out_json.exists():
        try:
            return json.load(out_json.open("r", encoding="utf-8"))
        except Exception:
            return []
    return []


def call_gpt_vision(model: str, api_key: str, image_b64: str, prompt_text: str | None = None) -> Dict[str, Any]:
    """Call GPT-4o (or compatible) with a text+image prompt; return parsed JSON.

    Assumes OpenAI-compatible HTTP API via openai>=1.x.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Please `pip install openai>=1.0.0`.") from e

    client = OpenAI(api_key=api_key)

    # Build message with image content part
    if prompt_text is None:
        prompt_text = STRICT_PROMPT_TEXT
    content = [
        {"type": "text", "text": prompt_text}
    ]
    content.append({
        "type": "image_url",
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


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_dedup(boxes: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda d: float(d.get("confidence", 0.5)), reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in boxes_sorted:
        cb = cand["bbox"]
        suppressed = False
        for k in kept:
            if iou_xyxy(cb, k["bbox"]) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


def generate_tiles(w: int, h: int, max_tile: int = 1024, overlap: float = 0.2) -> List[tuple[int, int, int, int]]:
    """Return list of (x1,y1,x2,y2) tile boxes in pixel coordinates."""
    # Ensure at least one tile
    if w <= max_tile and h <= max_tile:
        return [(0, 0, w, h)]
    sx = max_tile
    sy = max_tile
    stride_x = max(1, int(sx * (1 - overlap)))
    stride_y = max(1, int(sy * (1 - overlap)))
    tiles = []
    y = 0
    while True:
        x = 0
        y2 = min(h, y + sy)
        while True:
            x2 = min(w, x + sx)
            tiles.append((x, y, x2, y2))
            if x2 >= w:
                break
            x = x + stride_x
        if y2 >= h:
            break
        y = y + stride_y
    return tiles


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
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skip unreadable image: {img_path} ({e})")
            continue

        W, H = img.size

        # Tiled pass with overlap
        tiles = generate_tiles(W, H, max_tile=1024, overlap=0.2)
        all_items: List[Dict[str, Any]] = []

        for (x1p, y1p, x2p, y2p) in tiles:
            crop = img.crop((x1p, y1p, x2p, y2p))
            tile_b64 = encode_pil_to_base64(crop)
            result = call_gpt_vision(model=model, api_key=api_key, image_b64=tile_b64, prompt_text=STRICT_PROMPT_TEXT)
            items_tile = sanitize_items(result.get("items", []))
            # Map tile-normalized coords to global normalized
            tw = max(1, (x2p - x1p))
            th = max(1, (y2p - y1p))
            for it in items_tile:
                x1, y1, x2, y2 = it["bbox"]
                gx1 = (x1p + x1 * tw) / W
                gy1 = (y1p + y1 * th) / H
                gx2 = (x1p + x2 * tw) / W
                gy2 = (y1p + y2 * th) / H
                it["bbox"] = [clamp_float(gx1), clamp_float(gy1), clamp_float(gx2), clamp_float(gy2)]
                all_items.append(it)

        # If detections are few, run focused-region passes (bands)
        if len(all_items) < 10:
            bands = [
                (0, 0, W, max(int(0.10 * H), 1)),  # top 10%
                (0, H - max(int(0.10 * H), 1), W, H),  # bottom 10%
                (0, 0, max(int(0.12 * W), 1), H),  # left 12%
                (W - max(int(0.12 * W), 1), 0, W, H),  # right 12%
            ]
            for (x1p, y1p, x2p, y2p) in bands:
                crop = img.crop((x1p, y1p, x2p, y2p))
                band_b64 = encode_pil_to_base64(crop)
                result = call_gpt_vision(model=model, api_key=api_key, image_b64=band_b64, prompt_text=STRICT_PROMPT_TEXT)
                items_band = sanitize_items(result.get("items", []))
                tw = max(1, (x2p - x1p))
                th = max(1, (y2p - y1p))
                for it in items_band:
                    x1, y1, x2, y2 = it["bbox"]
                    gx1 = (x1p + x1 * tw) / W
                    gy1 = (y1p + y1 * th) / H
                    gx2 = (x1p + x2 * tw) / W
                    gy2 = (y1p + y2 * th) / H
                    it["bbox"] = [clamp_float(gx1), clamp_float(gy1), clamp_float(gx2), clamp_float(gy2)]
                    all_items.append(it)

        # Deduplicate with simple NMS
        deduped = nms_dedup(all_items, iou_thresh=0.5)

        if not deduped:
            print(f"No items for {img_path.name}")
            continue

        for it in deduped:
            existing.append({
                "image_path": str(img_path),
                "content": it["label"],
                "bbox": it["bbox"],
            })
            added += 1

        # Persist progressively to avoid data loss
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        print(f"Labeled {img_path.name}: +{len(deduped)} entries (total so far: {len(existing)})")

    print(f"\nDone. Added {added} entries. Output: {out_json}")


if __name__ == "__main__":
    main()


