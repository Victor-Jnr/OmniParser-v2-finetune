import torch
print(torch.version.cuda)  # 确认 CUDA 版本
print(torch.cuda.is_available())  # 检查是否支持 CUDA
print(torch.cuda.get_device_properties(0).total_memory / (1024**3)) # 打印 GPU 总内存