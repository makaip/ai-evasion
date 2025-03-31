import os
import random
import re
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from nltk.corpus import gutenberg, words
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
import logging
import wikipedia
from torch.amp import autocast, GradScaler
import spacy
from textblob import TextBlob

# Set environment variable for CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('words')

nlp = spacy.load("en_core_web_sm")

def extract_segments(text, min_tokens=10, max_tokens=20):
    tokens = word_tokenize(text)
    return [" ".join(tokens[i:i+max_tokens]) for i in range(len(tokens) - max_tokens) if len(tokens[i:i+max_tokens]) >= min_tokens]

def load_gutenberg_texts():
    texts = []
    for fileid in gutenberg.fileids():
        raw_text = gutenberg.raw(fileid)
        texts.extend(extract_segments(raw_text))
    return texts

def load_wikipedia_texts(num_articles=10):
    texts = []
    try:
        titles = wikipedia.random(num_articles)
        titles = [titles] if not isinstance(titles, list) else titles
        for title in titles:
            try:
                page = wikipedia.page(title)
                texts.extend(extract_segments(page.content))
            except:
                pass
    except:
        pass
    return texts

def corrupt_text(text):
    methods = [syntax_break, semantic_drift, tense_shift, nonsense_insert, reorder_structure, backtranslation]
    return random.choice(methods)(text)

def syntax_break(text):
    words_list = text.split()
    if len(words_list) > 3:
        idx = random.randint(1, len(words_list) - 2)
        words_list[idx] = words_list[idx][:len(words_list[idx])//2] + " " + words_list[idx][len(words_list[idx])//2:]
    return " ".join(words_list)

def semantic_drift(text):
    words_list = text.split()
    corpus_words = [w.lower() for w in words.words() if w.isalpha() and len(w) > 2]
    if words_list:
        words_list[random.randint(0, len(words_list)-1)] = random.choice(corpus_words)
    return " ".join(words_list)

def tense_shift(text):
    """
    Convert present-tense verbs to past tense using TextBlob.
    """
    blob = TextBlob(text)
    converted = []
    
    for word, tag in blob.tags:
        if tag.startswith("VB"):  # Identifies verbs
            past_tense = word + "ed" if not word.endswith("e") else word + "d"  # Basic heuristic
            converted.append(past_tense)
        else:
            converted.append(word)
    
    return " ".join(converted)

def nonsense_insert(text):
    words_list = text.split()
    words_list.insert(random.randint(0, len(words_list)), " ".join(random.choices(words.words(), k=random.randint(3, 7))))
    return " ".join(words_list)

def reorder_structure(text):
    words_list = text.split()
    return " ".join(words_list[len(words_list)//2:] + words_list[:len(words_list)//2]) if len(words_list) >= 6 else text

def backtranslation(text):
    return text  # Placeholder for future backtranslation logic

class CoherenceDataset(Dataset):
    def __init__(self, coherent_texts, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.samples = [(text, 1) for text in coherent_texts] + [(corrupt_text(text), 0) for text in coherent_texts]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

def train_model(model, dataset, epochs=10, batch_size=1, learning_rate=2e-5, patience=3):
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))
    scaler = GradScaler()
    best_val_acc, epochs_without_improvement = 0.0, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()
        logging.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {total_loss/len(train_loader):.4f}")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered.")
                break
        torch.cuda.empty_cache()

def main():
    MODEL_NAME = "/mnt/beegfs/home/jpindell2022/scratch/models/Opt-2.7b-HF"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    coherent_texts = random.sample(load_gutenberg_texts() + load_wikipedia_texts(), min(1000, len(load_gutenberg_texts())))
    dataset = CoherenceDataset(coherent_texts, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    train_model(model, dataset, epochs=10, batch_size=1, learning_rate=2e-5, patience=3)
    model.save_pretrained("coherence_model_opt2.7b")
    tokenizer.save_pretrained("coherence_model_opt2.7b")
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
