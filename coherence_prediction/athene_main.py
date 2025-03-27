import random
import re
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from nltk.corpus import gutenberg, words
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
import logging

# Set up logging for detailed experiment-level information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Download required NLTK resources
# ----------------------------
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('words')

# ----------------------------
# Data Preparation Functions
# ----------------------------

def extract_segments(text, min_tokens=10, max_tokens=20):
    """
    Extract contiguous token segments from a given text.
    """
    tokens = word_tokenize(text)
    segments = []
    for i in range(len(tokens) - max_tokens):
        seg = tokens[i:i+max_tokens]
        if len(seg) >= min_tokens:
            segments.append(" ".join(seg))
    return segments

def load_gutenberg_texts():
    """
    Load texts from the Gutenberg corpus and extract segments.
    """
    texts = []
    for fileid in gutenberg.fileids():
        logging.info(f"Processing Gutenberg file: {fileid}...")
        raw_text = gutenberg.raw(fileid)
        segments = extract_segments(raw_text)
        texts.extend(segments)
    return texts

def load_wikipedia_texts():
    """
    Placeholder for loading texts from Wikipedia.
    You can use libraries like 'wikipedia' or process a Wikipedia dump.
    For now, this returns an empty list.
    """
    logging.info("Loading Wikipedia texts... (this is a placeholder)")
    texts = []
    # TODO: Implement Wikipedia text extraction and segmentation.
    return texts

def load_additional_datasets():
    """
    Combine texts from multiple sources for a larger dataset.
    """
    texts = load_gutenberg_texts()
    texts += load_wikipedia_texts()
    # Add more sources as needed.
    logging.info(f"Total dataset size: {len(texts)} segments.")
    return texts

# ----------------------------
# Corruption Model: Negative Sample Generation
# ----------------------------

def syntax_break(text):
    words_list = text.split()
    if len(words_list) > 3:
        idx = random.randint(1, len(words_list)-2)
        word = words_list[idx]
        split_point = max(1, len(word) // 2)
        words_list[idx] = word[:split_point] + " " + word[split_point:]
    return " ".join(words_list)

def semantic_drift(text):
    words_list = text.split()
    if not words_list:
        return text
    idx = random.randint(0, len(words_list)-1)
    corpus_words = [w.lower() for w in words.words() if w.isalpha() and len(w) > 2]
    replacement = random.choice(corpus_words)
    words_list[idx] = replacement
    return " ".join(words_list)

def tense_shift(text):
    text = re.sub(r'\b(is|are|am)\b', 'was', text)
    text = re.sub(r'\b(runs)\b', 'ran', text)
    text = re.sub(r'\b(eats)\b', 'ate', text)
    return text

def nonsense_insert(text, num_words_range=(3, 7)):
    words_list = text.split()
    num_words = random.randint(*num_words_range)
    corpus_words = [w.lower() for w in words.words() if w.isalpha() and len(w) > 2]
    inserted_words = " ".join(random.choices(corpus_words, k=num_words))
    idx = random.randint(0, len(words_list))
    words_list.insert(idx, inserted_words)
    return " ".join(words_list)

def reorder_structure(text):
    words_list = text.split()
    if len(words_list) < 6:
        return text
    mid = len(words_list) // 2
    return " ".join(words_list[mid:] + words_list[:mid])

def backtranslation(text):
    """
    Placeholder for backtranslation.
    In practice, you'd use an API or pre-trained model to translate text to another language and back.
    """
    # TODO: Integrate with a translation API or model.
    return text  # For now, no change is made.

def corrupt_text(text):
    """
    Apply one of several corruption strategies.
    """
    corruption_methods = [syntax_break, semantic_drift, tense_shift, nonsense_insert, reorder_structure, backtranslation]
    method = random.choice(corruption_methods)
    return method(text)

# ----------------------------
# Dataset Definition
# ----------------------------

class CoherenceDataset(Dataset):
    """
    Dataset for Predictive Coherence Assessment.
    Each sample consists of a text segment and a binary label:
        1 for naturally coherent text,
        0 for synthetically corrupted text.
    """
    def __init__(self, coherent_texts, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Positive examples: naturally coherent segments (label = 1)
        for text in coherent_texts:
            self.samples.append((text, 1))
        
        # Negative examples: synthetically corrupted segments (label = 0)
        for text in coherent_texts:
            corrupted = corrupt_text(text)
            self.samples.append((corrupted, 0))
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ----------------------------
# Model Training Function with Early Stopping & Scheduler
# ----------------------------

def train_model(model, dataset, epochs=30, batch_size=2, learning_rate=2e-5, patience=3):
    """
    Fine-tune a pretrained transformer model for the coherence classification task.
    Note: Due to the size of LLaMA 8B, use a small batch size and consider gradient accumulation.
    """
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch["labels"]
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        f1 = f1_score(all_labels, all_preds, average='binary')
        logging.info(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}, F1 Score: {f1:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered.")
                break

# ----------------------------
# Main Execution
# ----------------------------

def main():
    """
    Main execution function:
      1. Loads and processes texts from multiple datasets.
      2. Constructs the dataset with both coherent and corrupted samples.
      3. Fine-tunes a pretrained LLaMA model on the dataset.
      4. Saves the fine-tuned model for later use.
    """
    # Download additional NLTK resources if necessary
    nltk.download('punkt')
    nltk.download('gutenberg')
    nltk.download('words')

    logging.info("Loading and processing texts from multiple datasets...")
    coherent_texts = load_additional_datasets()
    # Optionally sample a subset for efficiency; increase sample size for full-scale experiments.
    sample_size = min(1000, len(coherent_texts))
    coherent_texts = random.sample(coherent_texts, sample_size)
    
    # Set the model name to your LLaMA 8B (3.1) checkpoint.
    MODEL_NAME = "/mnt/onefs/scratch/jpindell2022/models/Llama-3.1-8B-HF"  # Update this with your actual model name or path
    
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = CoherenceDataset(coherent_texts, tokenizer, max_length=128)
    
    # Initialize model for binary sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Optional: Enable gradient checkpointing to save memory during fine-tuning.
    model.gradient_checkpointing_enable()
    
    # Train the model
    # Note: Batch size is set low due to the large size of LLaMA 8B.
    train_model(model, dataset, epochs=30, batch_size=2, learning_rate=2e-5, patience=3)
    
    # Save the fine-tuned model and tokenizer for reproducibility and future use
    model_save_path = "unsupervised_coherence_llama8b"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Training complete. Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()
