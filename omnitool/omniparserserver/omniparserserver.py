'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05

python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --port 9333
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
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--max_batch_size', type=int, default=256, help='Maximum batch size for processing')
    parser.add_argument('--enable_gpu_optimization', action='store_true', help='Enable GPU optimizations')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)
app = FastAPI()
omniparser = Omniparser(config)

def optimize_torch_settings():
    """Optimize PyTorch settings for better performance"""
    # Enable optimized attention if available
    if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set number of threads for CPU operations
    cpu_count = mp.cpu_count()
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)
    
    # Enable JIT compilation for better performance
    torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])

def initialize_models(config):
    """Initialize models with optimizations"""
    global omniparser
    
    # Optimize torch settings
    optimize_torch_settings()
    
    # Initialize omniparser with optimized config
    optimized_config = config.copy()
    if config.get('enable_gpu_optimization') and torch.cuda.is_available():
        optimized_config['batch_size'] = config.get('max_batch_size', 256)
        # Enable mixed precision if GPU is available
        torch.backends.cuda.enable_flash_sdp(True)
    
    omniparser = Omniparser(optimized_config)
    print(f'Omniparser initialized with device: {config["device"]}')
    print(f'PyTorch threads: {torch.get_num_threads()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global executor
    args = parse_arguments()
    config = vars(args)
    
    # Determine optimal number of workers
    if config['workers'] is None:
        if torch.cuda.is_available():
            # For GPU, use fewer workers to avoid memory conflicts
            config['workers'] = min(4, mp.cpu_count() // 2)
        else:
            # For CPU, use more workers
            config['workers'] = mp.cpu_count()
    
    print(f"Starting with {config['workers']} workers")
    
    # Initialize thread pool executor for CPU-bound tasks
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config['workers'],
        thread_name_prefix="omniparser"
    )
    
    # Initialize models
    initialize_models(config)
    
    yield
    
    # Shutdown
    if executor:
        executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

class ParseRequest(BaseModel):
    base64_image: str
    output_base64: bool = False

async def parse_image_async(image_base64: str):
    """Async wrapper for image parsing"""
    loop = asyncio.get_event_loop()
    
    # Run the CPU-intensive parsing in thread pool
    result = await loop.run_in_executor(
        executor, 
        omniparser.parse, 
        image_base64
    )
    return result

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    
    try:
        # Use async processing to avoid blocking
        dino_labled_img, parsed_content_list = await parse_image_async(parse_request.base64_image)
        
        latency = time.time() - start
        print(f'parsing completed in {latency:.2f}s')
        
        if parse_request.output_base64:
            return {
                "som_image_base64": dino_labled_img, 
                "parsed_content_list": parsed_content_list, 
                'latency': latency
            }
        else:
            return {
                "parsed_content_list": parsed_content_list, 
                'latency': latency
            }
    except Exception as e:
        print(f'Error during parsing: {str(e)}')
        return {
            "error": str(e),
            "parsed_content_list": [],
            'latency': time.time() - start
        }

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

@app.get("/health/")
async def health():
    """Health check endpoint with system info"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_threads": torch.get_num_threads(),
        "workers": executor._max_workers if executor else 0
    }

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)