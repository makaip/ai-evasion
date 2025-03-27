#!/bin/bash
#SBATCH --job-name=incoherent_ai_llama
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --partition=shortq7-gpu
#SBATCH --gres=gpu:3

scontrol show job $SLURM_JOB_ID

module load anaconda3-2021.05-gcc-9.4.0-llhdqho
module load cudnn-8.2.0.53-11.3-gcc-9.2.0-swfu6w4
module load cuda-11.7.1-gcc-9.4.0-jjbjea7

ENV_NAME="aidetection"

conda activate "$ENV_NAME"

echo "Installed package versions:"
python -c "import torch, transformers, nltk, numpy, sklearn, tqdm; \
           print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, \
           NLTK: {nltk.__version__}, NumPy: {numpy.__version__}, \
           Scikit-learn: {sklearn.__version__}, tqdm: {tqdm.__version__}')"

echo "Checking cuDNN installation..."
ls -l /usr/local/cuda/lib64/libcudnn*
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'cuDNN Available: {torch.backends.cudnn.is_available()}')"

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; \
           print(f'CUDA Available: {torch.cuda.is_available()}'); \
           print(f'GPU Count: {torch.cuda.device_count()}'); \
           print(f'Current GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected.')"

# Run the Python script with error handling
echo "Running Python script..."

python3 /home/jpindell2022/aidetection/coherence_prediction/caai_main.py
status=$?

if [ $status -ne 0 ]; then
    echo "Operation failed with exit code $status"
    exit $status
else
    echo "Script completed successfully!"
fi