import os
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Define dataset class
class CoherenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Setup logging
log_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/train_roberta.log"
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

# Load model and tokenizer
model_name = "roberta-base"  # Smaller than GPT2-medium but very effective
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type="single_label_classification"
).to(device)

if world_size > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

# Load cached dataset
dataset_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/cached_dataset.pkl"
logging.info(f"Rank {rank}: Loading dataset from {dataset_path}")

with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

# Function for preparing dataset splits with stratification
def prepare_dataset_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Assumes dataset items are tuples of (input_dict, label)
    # Extract labels for stratification
    try:
        labels = [item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in dataset]
    except:
        # If extraction fails, fall back to random split
        logging.warning(f"Rank {rank}: Failed to extract labels for stratification, using random split")
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Get indices for each class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Create stratified splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        n_val = int(val_ratio * len(indices))
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    logging.info(f"Rank {rank}: Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# Training function with validation and test evaluation
def train_model(model, dataset, epochs=5, batch_size=16, learning_rate=2e-5):
    # Split dataset with stratification
    train_dataset, val_dataset, test_dataset = prepare_dataset_splits(dataset)

    # Create data loaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer with weight decay and learning rate
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )

    # Loss function and mixed precision
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Early stopping parameters
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs, labels = batch
                
                # Move data to device
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                labels = labels.to(device, non_blocking=True)

                # Forward pass with mixed precision
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Track predictions
                _, preds = torch.max(outputs.logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()

                # Log progress
                if batch_idx % 20 == 0:
                    logging.info(f"Rank {rank} - Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            except Exception as e:
                logging.error(f"Rank {rank} - Error processing batch {batch_idx}: {str(e)}")
                continue

        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average='binary', zero_division=0
        )
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for val_batch in val_loader:
                try:
                    val_inputs, val_target_labels = val_batch
                    val_inputs = {k: v.to(device, non_blocking=True) for k, v in val_inputs.items()}
                    val_target_labels = val_target_labels.to(device, non_blocking=True)
                    
                    # Forward pass
                    with autocast(device_type='cuda', dtype=torch.float16):
                        val_outputs = model(**val_inputs)
                        batch_val_loss = loss_fn(val_outputs.logits, val_target_labels)
                    
                    val_loss += batch_val_loss.item()
                    
                    # Track predictions
                    _, val_batch_preds = torch.max(val_outputs.logits, dim=1)
                    val_preds.extend(val_batch_preds.cpu().numpy())
                    val_labels.extend(val_target_labels.cpu().numpy())
                
                except Exception as e:
                    logging.error(f"Rank {rank} - Error processing validation batch: {str(e)}")
                    continue
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', zero_division=0
        )
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch results
        logging.info(f"Rank {rank} - Epoch {epoch+1}:")
        logging.info(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        logging.info(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            
            # Save best model state
            if rank == 0:
                if hasattr(model, 'module'):
                    best_model_state = model.module.state_dict()
                else:
                    best_model_state = model.state_dict()
                
                save_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/best_roberta_model.pt"
                torch.save(best_model_state, save_path)
                logging.info(f"Best model saved with validation accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # After training is done, evaluate on the test set using the best model
    if rank == 0 and best_model_state is not None:
        # Create a test data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load the best model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        
        # Test evaluation
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for test_batch in test_loader:
                try:
                    test_inputs, test_target_labels = test_batch
                    test_inputs = {k: v.to(device, non_blocking=True) for k, v in test_inputs.items()}
                    test_target_labels = test_target_labels.to(device, non_blocking=True)
                    
                    # Forward pass
                    test_outputs = model(**test_inputs)
                    
                    # Track predictions
                    _, test_batch_preds = torch.max(test_outputs.logits, dim=1)
                    test_preds.extend(test_batch_preds.cpu().numpy())
                    test_labels.extend(test_target_labels.cpu().numpy())
                
                except Exception as e:
                    logging.error(f"Error processing test batch: {str(e)}")
                    continue
        
        # Calculate test metrics
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary', zero_division=0
        )
        
        # Log test results
        logging.info(f"Test Results with Best Model:")
        logging.info(f"  Accuracy: {test_accuracy:.4f}")
        logging.info(f"  Precision: {test_precision:.4f}")
        logging.info(f"  Recall: {test_recall:.4f}")
        logging.info(f"  F1 Score: {test_f1:.4f}")
        
    # Cleanup
    if world_size > 1:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logging.error(f"Rank {rank} - Error during cleanup: {str(e)}")

# Function to analyze model mistakes
def analyze_errors(model, dataset, tokenizer, batch_size=8):
    if rank == 0:  # Only run on main process
        # Create a simple dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        mistakes = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                outputs = model(**inputs)
                _, preds = torch.max(outputs.logits, dim=1)
                
                # Find mistakes
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        # Get the input text
                        if 'input_ids' in inputs:
                            input_text = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
                            mistakes.append({
                                'text': input_text,
                                'true_label': labels[i].item(),
                                'pred_label': preds[i].item(),
                                'confidence': torch.softmax(outputs.logits[i], dim=0)[preds[i]].item()
                            })
        
        # Log some of the mistakes for analysis
        if mistakes:
            logging.info(f"Found {len(mistakes)} mistakes. Showing first 5:")
            for i, mistake in enumerate(mistakes[:5]):
                logging.info(f"Mistake {i+1}:")
                logging.info(f"  Text: {mistake['text'][:100]}...")
                logging.info(f"  True label: {mistake['true_label']}, Pred label: {mistake['pred_label']}")
                logging.info(f"  Confidence: {mistake['confidence']:.4f}")
            
            # Save mistakes for further analysis
            with open("/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/model_mistakes.pkl", "wb") as f:
                pickle.dump(mistakes, f)

# Hyperparameter search function
def hyperparameter_search(model_class, dataset, tokenizer, param_grid, search_iterations=5):
    if rank == 0:  # Only run on main process
        # Split dataset
        train_dataset, val_dataset, _ = prepare_dataset_splits(dataset, train_ratio=0.8, val_ratio=0.2, test_ratio=0)
        
        # Create validation dataloader
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        best_params = None
        best_val_acc = 0
        
        # Log search start
        logging.info("Starting hyperparameter search:")
        logging.info(f"Parameter grid: {param_grid}")
        
        for i in range(search_iterations):
            # Sample random parameters
            params = {k: np.random.choice(v) for k, v in param_grid.items()}
            logging.info(f"Trial {i+1}/{search_iterations}: {params}")
            
            # Initialize model with current params
            model = model_class.from_pretrained(
                "roberta-base",
                num_labels=2,
                problem_type="single_label_classification"
            ).to(device)
            
            # Train with subset of data (quick training)
            subset_size = min(2000, len(train_dataset))
            subset_indices = random.sample(range(len(train_dataset)), subset_size)
            train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
            
            train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
            
            # Optimizer
            optimizer = AdamW(model.parameters(), lr=params['learning_rate'], eps=1e-8, 
                              weight_decay=params['weight_decay'])
            
            # Training loop - simplified for search
            model.train()
            for _ in range(params['epochs']):
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(**inputs)
                    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_target_labels = val_batch
                    val_inputs = {k: v.to(device) for k, v in val_inputs.items()}
                    val_target_labels = val_target_labels.to(device)
                    
                    val_outputs = model(**val_inputs)
                    _, val_batch_preds = torch.max(val_outputs.logits, dim=1)
                    val_preds.extend(val_batch_preds.cpu().numpy())
                    val_labels.extend(val_target_labels.cpu().numpy())
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(val_labels, val_preds)
            logging.info(f"  Validation accuracy: {val_accuracy:.4f}")
            
            # Update best parameters
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_params = params
                logging.info(f"  New best parameters!")
        
        logging.info(f"Hyperparameter search completed.")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return best_params
    return None

# Main execution
try:
    # First run a brief hyperparameter search (if resources allow)
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'batch_size': [8, 16, 32],
        'weight_decay': [0.01, 0.05, 0.1],
        'epochs': [2, 3]  # Just for the search
    }
    
    # Run hyperparameter search if requested
    run_search = False  # Set to True to enable search
    best_params = None
    
    if run_search and rank == 0:
        logging.info("Running hyperparameter search...")
        best_params = hyperparameter_search(
            RobertaForSequenceClassification, 
            dataset, 
            tokenizer, 
            param_grid,
            search_iterations=3  # Limited for time constraints
        )
    
    # Train the model with best parameters or defaults
    if best_params:
        train_model(
            model, 
            dataset, 
            epochs=5,  # More epochs for full training
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate']
        )
    else:
        # Use sensible defaults
        train_model(
            model, 
            dataset, 
            epochs=5,
            batch_size=16,
            learning_rate=2e-5
        )
    
    # Analyze errors to understand model weaknesses
    if rank == 0:
        logging.info("Analyzing model errors...")
        analyze_errors(model, dataset, tokenizer)

except Exception as e:
    logging.error(f"Rank {rank} - Error during execution: {str(e)}")
    # Make sure to clean up
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except:
            pass
    raise