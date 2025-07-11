'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05

python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence_merged_8bit --device cuda --BOX_TRESHOLD 0.05 --port 9333
'''

import sys
import os
import time
import asyncio
import concurrent.futures
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn
import torch
import multiprocessing as mp
from contextlib import asynccontextmanager

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

# Global variables for model instances
omniparser = None
executor = None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence_merged_8bit', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--max_batch_size', type=int, default=256, help='Maximum batch size for processing')
    parser.add_argument('--enable_gpu_optimization', action='store_true', help='Enable GPU optimizations')
    parser.add_argument('--use_paddleocr', default=True, help='Use paddleocr for ocr')
    args = parser.parse_args()
    return args


args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return { "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    # uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=False)