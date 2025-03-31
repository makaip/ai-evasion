

# AI Detection

```
scontrol update JobId=95100 MailUser=jpindell2022@fau.edu MailType=ALL
```

```
ssh jpindell2022@athene-login.hpc.fau.edu
```

To run the stuff idk

```
clear && sbatch job_script.sh && echo -e "\n\n" && sleep 1 && squeue -u $USER && echo -e "\n\n" && sleep 1 && tail -f output.txt
sbatch job_script.sh
```

list available nodes

```
sinfo -N -l
sinfo -o "%P %N %G"
squeue -o "%.18i %.9P %.20j %.8u %.8T %.10M %.10l %.6D %R"
```

start an interactive session

```
srun --ntasks=1 --cpus-per-task=8 --mem=64G --time=06:00:00 --partition=shortq7-gpu --pty bash
```

cancel a job

```
scancel <JOBID>
```

get the statuses

```
squeue -u $USER
tail -f output.txt
tail -f error.txt
```


## Initial Setup

Load Modules

```
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4
```

Create Conda Environment*

```
conda init
conda create -n aidetection python=3.11
```

Activate Conda Environment

```
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh 
conda activate aidetection

```

Install Packages*

```
conda install pytorch numpy scikit-learn tqdm -c pytorch
pip install transformers nltk datasets && python -m nltk.downloader punkt words gutenberg
python -c "import nltk, torch, numpy, transformers; print('All packages installed successfully')"
```

## Troubleshooting

sbatch: error: Batch script contains DOS line breaks (\r\n)

```
sed -i 's/\r//' 
```

### Total Reset


```
conda env list
conda env remove --name aidetection
```

This will remove all cached files and potential corrupted packages.

```
conda clean --all
```


## Data Transfer

```
scp -r jpindell2022@athene-login.hpc.fau.edu:/mnt/beegfs/home/jpindell2022/scratch/models/Llama-3.1-8B-HF .

```


 idk asjflskdf

start a session on shortq7-gpu partition with 2 GPUs and 20GB of memory

```
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --mem=20G --time=06:00:00 --partition=shortq7-gpu --pty bash
```

get info on job 94566

```
scontrol show jobid -dd 94566
```

```
squeue --job 94823
squeue --priority | awk 'NR==1 || $1==94823 || NR==FNR{print}'
squeue --state=PENDING --sort=P | awk '{print $1}' | grep -B10000 94823 | wc -l
```


```

module load ollama/0.4.2-gcc-13.2.0-7tjvakl

ollama serve &
export OLLAMA_HOME=/mnt/beegfs/home/jpindell2022/scratch/ollama
ollama pull llama3.3:70b
ollama run llama3.3:70b


```