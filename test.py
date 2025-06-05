import torch
print(torch.version.cuda)  # 确认 CUDA 版本
print(torch.cuda.is_available())  # 检查是否支持 CUDA