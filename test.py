import torch
print(torch.version.cuda)  # 确认 CUDA 版本
print(torch.cuda.is_available())  # 检查是否支持 CUDA
print(torch.cuda.get_device_properties(0).total_memory / (1024**3)) # 打印 GPU 总内存

import json

quantized_config_path = "weights/icon_caption_florence_merged_8bit/config.json"
                        


# Extract quantization_config from the config if it exists
if 'quantization_config' in config:
    quantization_config = config['quantization_config']
    
    # Save extracted quantization_config to quantized_config.json
    with open(quantized_config_path, 'w') as f:
        json.dump(quantization_config, f, indent=2)