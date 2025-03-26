import random
import re
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from nltk.corpus import gutenberg, words
from nltk.tokenize import word_tokenize
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
    
    Parameters:
        text (str): The input text.
        min_tokens (int): Minimum token count for a segment.
        max_tokens (int): Maximum token count for a segment.
        
    Returns:
        segments (list of str): List of text segments.
    """
    tokens = word_tokenize(text)
    segments = []
    for i in range(len(tokens) - max_tokens):
        seg = tokens[i:i+max_tokens]
        if len(seg) >= min_tokens:
            segments.append(" ".join(seg))
    return segments

def load_nltk_texts():
    """
    Load texts from the Gutenberg corpus and extract segments.
    
    Returns:
        texts (list of str): List of text segments extracted from all Gutenberg texts.
    """
    texts = []
    for fileid in gutenberg.fileids():
        logging.info(f"Processing {fileid}...")
        raw_text = gutenberg.raw(fileid)
        segments = extract_segments(raw_text)
        texts.extend(segments)
    return texts

# ----------------------------
# Corruption Model: Negative Sample Generation
# ----------------------------

def syntax_break(text):
    """
    Introduce syntax errors by inserting spaces or splitting a word.
    """
    words_list = text.split()
    if len(words_list) > 3:
        idx = random.randint(1, len(words_list)-2)
        word = words_list[idx]
        # Break the word in half with an inserted space
        split_point = max(1, len(word)//2)
        words_list[idx] = word[:split_point] + " " + word[split_point:]
    return " ".join(words_list)

def semantic_drift(text):
    """
    Replace a word with another semantically unrelated word selected from the NLTK words corpus.
    """
    words_list = text.split()
    if not words_list:
        return text
    idx = random.randint(0, len(words_list)-1)
    # Select a random word from the words corpus (ensure lower case for consistency)
    corpus_words = [w.lower() for w in words.words() if w.isalpha() and len(w) > 2]
    replacement = random.choice(corpus_words)
    words_list[idx] = replacement
    return " ".join(words_list)

def tense_shift(text):
    """
    Heuristically alter verb tenses by replacing common present-tense forms with past-tense.
    For a more rigorous implementation, one might use NLP tools like spaCy.
    """
    text = re.sub(r'\b(is|are|am)\b', 'was', text)
    text = re.sub(r'\b(runs)\b', 'ran', text)
    text = re.sub(r'\b(eats)\b', 'ate', text)
    return text

def nonsense_insert(text, num_words_range=(3, 7)):
    """
    Insert a sequence of random words (sampled from the NLTK words corpus) into the text.
    This method uses a large vocabulary to reduce training bias.
    
    Parameters:
        text (str): The original text.
        num_words_range (tuple): Minimum and maximum number of random words to insert.
        
    Returns:
        str: Text with a random sequence of words inserted.
    """
    words_list = text.split()
    # Determine how many words to insert
    num_words = random.randint(*num_words_range)
    corpus_words = [w.lower() for w in words.words() if w.isalpha() and len(w) > 2]
    inserted_words = " ".join(random.choices(corpus_words, k=num_words))
    idx = random.randint(0, len(words_list))
    words_list.insert(idx, inserted_words)
    return " ".join(words_list)

def reorder_structure(text):
    """
    Reorder the structure by reversing two random contiguous parts of the text.
    """
    words_list = text.split()
    if len(words_list) < 6:
        return text
    # Split the text into two halves and swap them
    mid = len(words_list) // 2
    return " ".join(words_list[mid:] + words_list[:mid])

def corrupt_text(text):
    """
    Randomly apply one of several corruption strategies to a given text.
    The randomness across many strategies helps reduce bias.
    
    Parameters:
        text (str): The original text segment.
        
    Returns:
        str: The corrupted version of the text.
    """
    corruption_methods = [syntax_break, semantic_drift, tense_shift, nonsense_insert, reorder_structure]
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
    def __init__(self, coherent_texts, tokenizer, max_length=64):
        """
        Initialize the dataset.
        
        Parameters:
            coherent_texts (list of str): List of coherent text segments.
            tokenizer: Hugging Face tokenizer.
            max_length (int): Maximum token length for model inputs.
        """
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
        # Remove extra batch dimension from tokenizer output
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ----------------------------
# Model Training Function
# ----------------------------

def train_model(model, dataset, epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Fine-tune a pretrained transformer model for the coherence classification task.
    
    Parameters:
        model: Pretrained transformer model for sequence classification.
        dataset (CoherenceDataset): The dataset for training and validation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
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
    
    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Transfer batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        accuracy = correct / total
        logging.info(f"Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f}")

# ----------------------------
# Main Execution
# ----------------------------

def main():
    """
    Main execution function that:
      1. Loads and processes texts from NLTK’s Gutenberg corpus.
      2. Constructs the dataset with both coherent and corrupted samples.
      3. Fine-tunes a pretrained BERT model on the dataset.
      4. Saves the fine-tuned model for later use.
    """

    nltk.download('punkt')
    nltk.download('gutenberg')
    nltk.download('words')

    logging.info("Loading and processing texts from Gutenberg corpus...")
    coherent_texts = load_nltk_texts()
    # Optionally sample a subset for efficiency; increase sample size for full-scale experiments.
    sample_size = min(1000, len(coherent_texts))
    coherent_texts = random.sample(coherent_texts, sample_size)
    
    # Initialize tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoherenceDataset(coherent_texts, tokenizer, max_length=64)
    
    # Initialize model for binary sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Train the model
    train_model(model, dataset, epochs=30, batch_size=16)
    
    # Save the fine-tuned model and tokenizer for reproducibility and future use
    model_save_path = "unsupervised_coherence_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Training complete. Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()
