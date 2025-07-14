'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05

python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence_merged_8bit --device cuda --BOX_TRESHOLD 0.05 --port 9333
'''

import sys
import os
import time
import asyncio
import concurrent.futures
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import argparse
import uvicorn
import torch
import multiprocessing as mp
from contextlib import asynccontextmanager
import uuid
from typing import Dict, List
import base64

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
# 添加GZIP压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 存储分块传输的会话数据
chunk_sessions: Dict[str, Dict] = {}

omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

class ChunkInitRequest(BaseModel):
    total_chunks: int
    chunk_size: int
    file_size: int

class ChunkUploadRequest(BaseModel):
    session_id: str
    chunk_index: int
    chunk_data: str

class ChunkProcessRequest(BaseModel):
    session_id: str

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return { "parsed_content_list": parsed_content_list, 'latency': latency}

@app.post("/parse/chunk/init/")
async def init_chunk_upload(request: ChunkInitRequest):
    """初始化分块上传会话"""
    session_id = str(uuid.uuid4())
    chunk_sessions[session_id] = {
        'total_chunks': request.total_chunks,
        'chunk_size': request.chunk_size,
        'file_size': request.file_size,
        'chunks': {},
        'created_at': time.time()
    }
    return {'session_id': session_id}

@app.post("/parse/chunk/upload/")
async def upload_chunk(request: ChunkUploadRequest):
    """上传单个数据块"""
    if request.session_id not in chunk_sessions:
        return {'error': 'Session not found'}
    
    session = chunk_sessions[request.session_id]
    session['chunks'][request.chunk_index] = request.chunk_data
    
    uploaded_chunks = len(session['chunks'])
    total_chunks = session['total_chunks']
    
    return {
        'uploaded_chunks': uploaded_chunks,
        'total_chunks': total_chunks,
        'complete': uploaded_chunks == total_chunks
    }

@app.post("/parse/chunk/process/")
async def process_chunked_image(request: ChunkProcessRequest):
    """处理分块上传的图像"""
    if request.session_id not in chunk_sessions:
        return {'error': 'Session not found'}
    
    session = chunk_sessions[request.session_id]
    
    # 检查是否所有块都已上传
    if len(session['chunks']) != session['total_chunks']:
        return {'error': 'Not all chunks uploaded'}
    
    # 重组base64数据
    base64_parts = []
    for i in range(session['total_chunks']):
        if i not in session['chunks']:
            return {'error': f'Missing chunk {i}'}
        base64_parts.append(session['chunks'][i])
    
    complete_base64 = ''.join(base64_parts)
    
    # 清理会话数据
    del chunk_sessions[request.session_id]
    
    # 处理图像
    print('start parsing chunked image...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(complete_base64)
    latency = time.time() - start
    print('time:', latency)
    
    return {'parsed_content_list': parsed_content_list, 'latency': latency}

@app.post("/parse/file/")
async def parse_file(file: UploadFile = File(...)):
    """接收文件上传并处理"""
    print('start parsing uploaded file...')
    start = time.time()
    
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 转换为base64
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        # 处理图像
        dino_labled_img, parsed_content_list = omniparser.parse(base64_image)
        latency = time.time() - start
        print('time:', latency)
        
        return {'parsed_content_list': parsed_content_list, 'latency': latency}
        
    except Exception as e:
        return {'error': str(e)}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    # uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=False)