import os
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import time
import logging

# Setup logging
log_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/train.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize distributed training
def setup_distributed():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

rank, local_rank, world_size = setup_distributed()

# GPU setup
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
logging.info(f'Rank {rank}: Using device: {device}')

# Load model
model_name = "/mnt/beegfs/home/jpindell2022/scratch/models/Opt-2.7b-HF"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

if world_size > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)
scaler = GradScaler(device='cuda')

# Load cached dataset
DATASET_PATH = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/cached_dataset.pkl"
logging.info(f"Rank {rank}: Loading dataset from {DATASET_PATH}")

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

def train_model(model, dataset, epochs=3, batch_size=32):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

        logging.info(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader):.4f}")

train_model(model, dataset, epochs=3, batch_size=32)
