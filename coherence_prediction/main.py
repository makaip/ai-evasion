import os
import random
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from GPUtil import showUtilization
import time
import sys

# Import text processing functions
from processing import load_gutenberg_texts, load_wikipedia_texts, corrupt_text

# Suppress BeautifulSoup warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)



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
num_gpus = torch.cuda.device_count()
print(f'Rank {rank}: Using device: {device}, GPU count: {num_gpus}')

# Load model and tokenizer (Using Fast Tokenizer)
model_name = "/mnt/beegfs/home/jpindell2022/scratch/models/Opt-2.7b-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    ).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    return model

model = get_model()
optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)
scaler = torch.amp.GradScaler(device='cuda')  # Updated GradScaler

class CoherenceDataset(Dataset):
    def __init__(self, coherent_texts):
        print("Preprocessing and caching dataset...")
        self.data = []

        # Precompute corrupted texts to avoid runtime overhead
        corrupted_texts = [corrupt_text(text) for text in coherent_texts]

        for text, label in zip(coherent_texts + corrupted_texts, [1] * len(coherent_texts) + [0] * len(corrupted_texts)):
            encoded = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            self.data.append(({key: val.squeeze(0) for key, val in encoded.items()}, torch.tensor(label, dtype=torch.long)))

        print(f"Created dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(model, dataset, epochs=3, batch_size=32):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset) if world_size > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()  # Track CPU-to-GPU transfer time

            inputs, labels = batch
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            # Debugging
            if batch_idx == 0 and rank == 0:
                print(f"Input tensor device: {inputs['input_ids'].device}")
                print(f"Model device: {next(model.parameters()).device}")
                showUtilization()

            with autocast():
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            # Track and print CPU-to-GPU transfer time
            transfer_time = time.time() - start_time
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f} | Transfer Time: {transfer_time:.4f}s")
                showUtilization()

        if rank == 0:
            print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                labels = labels.to(device, non_blocking=True)

                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if rank == 0:
            print(f"Validation Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    log_file = open(f"script_output_{rank}.txt", "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file  # Also capture errors

    print("Logging started...")  # First entry in log file

    if rank == 0:
        print("Loading datasets...")

    coherent_texts = load_gutenberg_texts() + load_wikipedia_texts(50)
    dataset = CoherenceDataset(coherent_texts)  # Now preprocessed and cached

    if rank == 0:
        print(f"Total dataset size: {len(dataset)}")
        print("Starting training...")

    train_model(model, dataset, epochs=3, batch_size=32)

    if rank == 0:
        print("Saving model...")
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained("coherence_model_opt2.7b")
        tokenizer.save_pretrained("coherence_model_opt2.7b")

    # Ensure all processes close the log file
    log_file.close()