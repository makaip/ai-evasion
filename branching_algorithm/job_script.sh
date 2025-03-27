#!/bin/bash
#SBATCH --job-name=ollama_sampling
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:30:00
#SBATCH --partition=shortq7-gpu
#SBATCH --gres=gpu:1

# Display job details
scontrol show job $SLURM_JOB_ID

# Load required modules
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4
module load ollama/0.4.2-gcc-13.2.0-7tjvakl
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k

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

# Run the Python script with error handling
echo "Running Python script..."
python3 /mnt/beegfs/home/jpindell2022/projects/aiouri/ai-evasion/coherence_prediction/main.py
status=$?

if [ $status -ne 0 ]; then
    echo "Operation failed with exit code $status"
    exit $status
else
    echo "Script completed successfully!"
fi
