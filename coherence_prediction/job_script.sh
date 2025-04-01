#!/bin/bash
#SBATCH --job-name=coherence_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=shortq7-gpu

# Load modules
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k

# Activate environment
ENV_NAME="aidetection"
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Run the training script
srun python /mnt/onefs/home/jpindell2022/projects/aiouri/ai-evasion/coherence_prediction/main.py