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

module load ollama/0.4.2-gcc-13.2.0-7tjvakl

# Persist environment variables
echo 'export OLLAMA_HOME="/mnt/beegfs/home/jpindell2022/scratch/ollama"' >> ~/.bashrc
echo 'export OLLAMA_MODELS="/mnt/beegfs/home/jpindell2022/scratch/ollama"' >> ~/.bashrc
source ~/.bashrc

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
sleep 5  # Give it some time to initialize

# Verify Ollama is running
echo "Checking if Ollama is running..."
python3 -c 'import requests; print("Ollama is running." if requests.get("http://localhost:11434/api/tags").status_code == 200 else "Ollama not responding.")'

ollama list

# Load required modules
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4

# Set environment name
ENV_NAME="aidetection"

# Activate the Conda environment
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Current Conda Environment: $CONDA_PREFIX"

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

# Run the Python script with error handling
echo "Running Python script..."
python3 /mnt/beegfs/home/jpindell2022/projects/aiouri/ai-evasion/branching_algorithm/main.py
status=$?

if [ $status -ne 0 ]; then
    echo "Operation failed with exit code $status"
    exit $status
else
    echo "Script completed successfully!"
fi