#!/bin/bash
#SBATCH --job-name=incoherent_ai        # Job name
#SBATCH --output=output.txt             # Standard output log
#SBATCH --error=error.txt               # Error log
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=64G                       # Memory allocation
#SBATCH --time=05:00:00                 # Time limit
#SBATCH --partition=shortq7-gpu         # GPU partition name (specific to your cluster)
#SBATCH --gres=gpu:1                    # Number of GPUs

scontrol show job $SLURM_JOB_ID

module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k

conda create -n aidetection python=3.11
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh
conda activate aidetection

conda install -y pytorch torchvision torchaudio -c pytorch
pip install -U transformers nltk numpy
python -m nltk.downloader punkt words gutenberg
python -c "import nltk, torch, numpy, transformers; print('All packages installed successfully')" 

# Debug: Show available GPUs
echo "Checking GPU availability..."
nvidia-smi

conda update numpy
conda update --all

# Debug: Verify PyTorch detects the GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Run the Python script
echo "Running python script"
python3 /mnt/beegfs/home/jpindell2022/projects/aiouri/ai-evasion/coherence_prediction/main.py || echo "Operation failed."

echo "All done"