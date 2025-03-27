#!/bin/bash
#SBATCH --job-name=ollama_sampling
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

module load anaconda/3/2019.03
module load cuda-11.7.1-gcc-9.4.0-jjbjea7
