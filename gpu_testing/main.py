import os
import torch

nvidia_smi_output = os.popen('nvidia-smi').read()
cuda_available = torch.cuda.is_available()
gpu_count = 0

if cuda_available:
    gpu_count = torch.cuda.device_count()

# Write the results to a file
with open('gpu_info.txt', 'w') as file:
    file.write("NVIDIA-SMI Output:\n")
    file.write(nvidia_smi_output + "\n")
    file.write(f"PyTorch CUDA Available: {cuda_available}\n")
    file.write(f"Number of GPUs accessible by CUDA: {gpu_count}\n")

