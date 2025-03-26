




```
sinfo -N -l
sinfo -o "%P %N %G"

```


```
clear && sbatch --nodelist=nodegpu031 job_script.sh && squeue -u $USER && echo "\n\n" && tail -f output.txt
```

```
srun --nodelist=nodegpu031 --ntasks=1 --cpus-per-task=8 --mem=64G --time=06:00:00 --partition=shortq7-gpu --pty bash
```


```
scancel <JOBID>
```


m progress

```
squeue -u $USER
tail -f output.txt
tail -f error.txt
```


```
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k

conda init
conda create -n aidetection python=3.11
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh 
conda activate aidetection

```

```
conda install -y pytorch torchvision torchaudio -c pytorch
pip install -U transformers nltk numpy && python -m nltk.downloader punkt words gutenberg
python -c "import nltk, torch, numpy, transformers; print('All packages installed successfully!')"

```


```
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k && conda init && source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh && conda activate aidetection
```