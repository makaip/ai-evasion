# AI Detection Guide  

## Initial Setup  

### Load Required Modules  
```  
module load cuda/12.4.0-gcc-13.2.0-shyinv2  
module load anaconda3/2023.09-0-gcc-13.2.0-dmzia4k  
module load cudnn/8.9.7.29-12-gcc-13.2.0-vpzj2v4  
```  

### Create and Activate Conda Environment  
```  
conda init  
conda create -n aidetection python=3.11  
```  

Activate the environment:  
```  
source /opt/ohpc/pub/spack/opt/spack/linux-rocky8-x86_64/gcc-13.2.0/anaconda3-2023.09-0-dmzia4k5kqs3plogxdfbu54jtqps54ma/etc/profile.d/conda.sh  
conda activate aidetection  
```  

### Install Required Packages  
```  
conda install pytorch numpy scikit-learn tqdm -c pytorch  
pip install transformers nltk datasets && python -m nltk.downloader punkt words gutenberg  
python -c "import nltk, torch, numpy, transformers; print('All packages installed successfully')"  
```  

---  

## Running Jobs on HPC  

### Connecting to the HPC Cluster  
```  
ssh jpindell2022@athene-login.hpc.fau.edu  
```  

### Submitting and Monitoring Jobs  
Submit a job:  
```  
sbatch job_script.sh  
```  

Monitor running jobs:  
```  
squeue -u $USER  
tail -f output.txt  
tail -f error.txt  
```  

Check job status:  
```  
scontrol show jobid -dd <JOBID>  
squeue --job <JOBID>  
squeue --priority | awk 'NR==1 || $1==<JOBID> || NR==FNR{print}'  
squeue --state=PENDING --sort=P | awk '{print $1}' | grep -B10000 <JOBID> | wc -l  
```  

Cancel a job:  
```  
scancel <JOBID>  
```  

---  

## Interactive Sessions  

Start an interactive session on `shortq7-gpu` with 8 CPUs, 64GB RAM:  
```  
srun --ntasks=1 --cpus-per-task=8 --mem=64G --time=06:00:00 --partition=shortq7-gpu --pty bash  
```  

Start a session with 2 GPUs and 20GB RAM:  
```  
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --mem=20G --time=06:00:00 --partition=shortq7-gpu --pty bash  
```  

---  

## Checking Available Resources  

List available nodes:  
```  
sinfo -N -l  
sinfo -o "%P %N %G"  
```  

View job queue:  
```  
squeue -o "%.18i %.9P %.20j %.8u %.8T %.10M %.10l %.6D %R"  
```  

---  

## Data Transfer  

Transfer files from HPC to local:  
```  
scp -r jpindell2022@athene-login.hpc.fau.edu:/mnt/beegfs/home/jpindell2022/scratch/models/Llama-3.1-8B-HF .  
```  

---  

## AI Model Setup with Ollama  

Load Ollama module:  
```  
module load ollama/0.4.2-gcc-13.2.0-7tjvakl  
```  

Start Ollama server and pull model:  
```  
ollama serve &  
export OLLAMA_HOME=/mnt/beegfs/home/jpindell2022/scratch/ollama  
ollama pull llama3.3:70b  
ollama run llama3.3:70b  
```  

---  

## Troubleshooting  

### Fixing DOS Line Breaks in Batch Script  
```  
sed -i 's/\r//' job_script.sh  
```  

### Reset Conda Environment  
List environments:  
```  
conda env list  
```  

Remove `aidetection` environment:  
```  
conda env remove --name aidetection  
```  

Clean all cached data:  
```  
conda clean --all  
```  
