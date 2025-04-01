import torch
import time
import pynvml
import multiprocessing
from datetime import datetime

def stress_test(device, duration=60):
    """Runs a stress test on a given GPU for a specified duration."""
    torch.cuda.set_device(device)
    size = (1024, 1024, 1024)  # Large tensor size to load GPU
    tensor = torch.randn(size, device=device)
    start_time = time.time()
    
    while time.time() - start_time < duration:
        tensor = tensor * 1.00001  # Keep the tensor busy
        tensor = tensor / 1.00001
        torch.cuda.synchronize()
    
    return f"GPU {device} stress test completed."

def log_gpu_utilization(interval=5, duration=60):
    """Logs GPU utilization every `interval` seconds for `duration` seconds."""
    pynvml.nvmlInit()
    log_file = "gpu_logs.txt"
    start_time = time.time()
    
    with open(log_file, "w") as f:
        while time.time() - start_time < duration:
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - GPU {i}: {utilization}%\n")
            time.sleep(interval)
    
    pynvml.nvmlShutdown()

def main():
    """Runs stress test and logs GPU utilization."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 4:
        print("Warning: Less than 4 GPUs detected!")
    
    # Start logging GPU utilization in a separate process
    log_process = multiprocessing.Process(target=log_gpu_utilization, args=(5, 60))
    log_process.start()
    
    # Start stress tests in parallel
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=stress_test, args=(i, 60))
        p.start()
        processes.append(p)
    
    # Wait for all stress tests to complete
    for p in processes:
        p.join()
    
    log_process.join()
    
    with open("results.txt", "w") as f:
        for i in range(num_gpus):
            f.write(f"GPU {i} stress test completed.\n")
    
    print("Stress test completed. Check results.txt and gpu_logs.txt for details.")

if __name__ == "__main__":
    main()
