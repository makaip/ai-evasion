




```
sinfo -N -l
sinfo -o "%P %N %G"

```


```
sbatch --nodelist=nodegpu031 job_script.sh
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
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
pip install -U transformers nltk numpy && python -m nltk.downloader punkt words gutenberg
python -c "import nltk, torch, numpy, transformers; print('All packages installed successfully!')"

```


```
```