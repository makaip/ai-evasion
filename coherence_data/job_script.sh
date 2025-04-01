#!/bin/bash

#SBATCH --job-name=coherence_preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=100G
#SBATCH --output=slurm-preprocess-%j.out
#SBATCH --error=slurm-preprocess-%j.err

# Load modules
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k

# Activate environment
ENV_NAME="aidetection"
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Run the preprocessing script
python /mnt/onefs/home/jpindell2022/projects/aiouri/ai-evasion/coherence_data/preprocess_data.py

echo "Preprocessing completed."
