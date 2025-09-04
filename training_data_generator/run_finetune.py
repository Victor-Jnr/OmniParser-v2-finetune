#!/usr/bin/env python3
"""
Wrapper to fine-tune Florence2 via the existing LoRA script on newly generated data.

Usage:
  python training_data_generator/run_finetune.py \
    --data ./training_data_generator/training_data/florence_format/florence_data.json \
    --model_path ./weights/icon_caption_florence \
    --epochs 25 --batch_size 8 --lr 5e-5
"""

import os
import sys
import argparse
import subprocess


def run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Florence2 LoRA fine-tuning")
    parser.add_argument("--data", default="./training_data_generator/training_data/florence_format/florence_data.json")
    parser.add_argument("--model_path", default="./weights/icon_caption_florence")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data not found: {args.data}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: model_path not found: {args.model_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "./finetune_omniparser_lora.py",
        "--data", args.data,
        "--model_path", args.model_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ]

    code = run(cmd)
    sys.exit(code)


if __name__ == "__main__":
    main()


