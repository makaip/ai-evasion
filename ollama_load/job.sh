#!/bin/bash

#SBATCH --job-name=ollama_sampling
#SBATCH --partition=shortq7-gpu   # Partition name
#SBATCH --gres=gpu:4              # Number of GPUs
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=64G                 # Memory allocation
#SBATCH --output=slurm-%j.out     # Standard output log
#SBATCH --error=slurm-%j.err      # Error log file

# Display job details
scontrol show job $SLURM_JOB_ID

# Load required modules
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4
module load ollama/0.4.2-gcc-13.2.0-7tjvakl

# Set environment name
ENV_NAME="aidetection"

# Activate the Conda environment
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Debug: Check installed package versions
echo "Installed package versions:"
python -c "import torch, json, random; \
           print(f'PyTorch: {torch.__version__}, JSON: {json.__name__}, Random: {random.__name__}')"

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; \
           print(f'CUDA Available: {torch.cuda.is_available()}'); \
           print(f'GPU Count: {torch.cuda.device_count()}'); \
           print(f'Current GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected.')"

echo "Installing Ollama thing idk brughsdhksdfj h"

export OLLAMA_HOME=/mnt/beegfs/home/jpindell2022/scratch/ollama
ollama pull llama3.3:70b
ollama run llama3.3:70b

ollama serve &

echo "it wokred"

delay 30

# Verify Ollama is running
echo "Checking if Ollama is running..."
python3 -c 'import requests; print("Ollama is running." if requests.get("http://localhost:11434/api/tags").status_code == 200 else "Ollama not responding.")'

