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
def train_model(model, dataset, epochs=5, batch_size=16):
    # Split dataset with stratification if possible
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Improved optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    
    # Better scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs, labels = batch
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    logging.info(f"Rank {rank} - Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: "
                                f"Loss = {loss.item():.4f}, Acc = {100 * correct / total:.2f}%")
            
            except Exception as e:
                logging.error(f"Rank {rank} - Error processing batch {batch_idx}: {str(e)}")
                continue

        # Validation after each epoch
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                try:
                    val_inputs, val_labels = val_batch
                    val_inputs = {k: v.to(device, non_blocking=True) for k, v in val_inputs.items()}
                    val_labels = val_labels.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        val_outputs = model(**val_inputs)
                        loss = loss_fn(val_outputs.logits, val_labels)
                    
                    val_loss += loss.item()
                    _, val_predicted = torch.max(val_outputs.logits, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()
                
                except Exception as e:
                    logging.error(f"Rank {rank} - Error processing validation batch: {str(e)}")
                    continue
        
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total if total > 0 else 0
        
        logging.info(f"Rank {rank} - Epoch {epoch+1}: "
                    f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.2f}%, "
                    f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.2f}%")
        
        # Early stopping with patience
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            
            # Save the best model
            if rank == 0:
                if hasattr(model, 'module'):
                    model_to_save = model.module
                else:
                    model_to_save = model
                
                save_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/best_model_checkpoint.pt"
                torch.save(model_to_save.state_dict(), save_path)
                logging.info(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save the final model (only by rank 0)
    if rank == 0:
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        
        save_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/final_model_checkpoint.pt"
        torch.save(model_to_save.state_dict(), save_path)
        logging.info(f"Final model saved to {save_path}")
        logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Cleanup
    if world_size > 1:
        try:
            dist.barrier()
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