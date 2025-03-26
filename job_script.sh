#!/bin/bash
#SBATCH --job-name=incoherent_ai        # Job name
#SBATCH --output=output.txt             # Standard output log
#SBATCH --error=error.txt               # Error log
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=64G                       # Memory allocation
#SBATCH --time=02:00:00                 # Time limit
#SBATCH --partition=shortq7-gpu         # GPU partition name (specific to your cluster)
#SBATCH --gres=gpu:1                    # Number of GPUs

# Load necessary modules
module load cuda/11.8

# Debug: Show available GPUs
echo "Checking GPU availability..."
nvidia-smi

# Debug: Verify PyTorch detects the GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Installing Nessecary Packages
pip install -U torch numpy transformers nltk

# Run the Python script
python ./coherence_prediction/main.py
