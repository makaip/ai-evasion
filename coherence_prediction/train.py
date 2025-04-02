import os
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import time
import logging

# Define dataset class
class CoherenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Setup logging
log_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/train.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize distributed training
def setup_distributed():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

rank, local_rank, world_size = setup_distributed()

# GPU setup
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
logging.info(f"Rank {rank}: Using device: {device}")

# Load model with padding token configuration
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Rank {rank}: Setting pad token to eos token: {tokenizer.pad_token}")

# Load configuration and update it
config = GPT2Config.from_pretrained(model_name, num_labels=2)
config.pad_token_id = tokenizer.pad_token_id

# Initialize model with the updated config
model = GPT2ForSequenceClassification.from_pretrained(
    model_name,
    config=config
).to(device)

if world_size > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

# Optimizer, scheduler, and mixed precision scaler
optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)
scaler = GradScaler()

# Load cached dataset
dataset_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/cached_dataset.pkl"
logging.info(f"Rank {rank}: Loading dataset from {dataset_path}")

with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

# Training function
def train_model(model, dataset, epochs=3, batch_size=32):
    # Try to catch and log the dataset structure
    try:
        sample_item = dataset[0]
        logging.info(f"Rank {rank}: Dataset sample structure: {type(sample_item)}")
        if isinstance(sample_item, tuple) and len(sample_item) == 2:
            logging.info(f"Rank {rank}: Input type: {type(sample_item[0])}, Labels type: {type(sample_item[1])}")
            if isinstance(sample_item[0], dict):
                logging.info(f"Rank {rank}: Input keys: {sample_item[0].keys()}")
    except Exception as e:
        logging.error(f"Rank {rank}: Error inspecting dataset: {str(e)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs, labels = batch
                
                # Debug the batch
                if batch_idx == 0:
                    logging.info(f"Rank {rank}: Batch structure - inputs type: {type(inputs)}, labels type: {type(labels)}")
                    if isinstance(inputs, dict):
                        logging.info(f"Rank {rank}: Input keys: {inputs.keys()}")
                        for k, v in inputs.items():
                            logging.info(f"Rank {rank}: {k} shape: {v.shape}")
                    logging.info(f"Rank {rank}: Labels shape: {labels.shape}")
                
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Use FP16 precision with device type specified
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                total_loss += loss.item()

                # Add progress logging for each batch
                if batch_idx % 10 == 0:  # Log every 10 batches
                    logging.info(f"Rank {rank} - Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            except Exception as e:
                logging.error(f"Rank {rank} - Error processing batch {batch_idx}: {str(e)}")
                # Continue with next batch instead of crashing
                continue

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Rank {rank} - Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # Save the model at the end of training (only by rank 0)
    if rank == 0:
        if hasattr(model, 'module'):
            model_to_save = model.module  # Get the model from the DDP wrapper
        else:
            model_to_save = model
        
        save_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/model_checkpoint.pt"
        torch.save(model_to_save.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")

    # Ensure proper cleanup
    if world_size > 1:
        try:
            dist.barrier()  # Synchronize before destroying process group
            dist.destroy_process_group()
        except Exception as e:
            logging.error(f"Rank {rank} - Error during cleanup: {str(e)}")

try:
    # Use a small batch size to avoid other potential issues
    train_model(model, dataset, epochs=3, batch_size=8)
except Exception as e:
    logging.error(f"Rank {rank} - Training failed with error: {str(e)}")
    # Make sure to clean up even if there's an error
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except:
            pass
    raise  # Re-raise to see the full error stack