import os
import random
import re
import nltk
import spacy
from textblob import TextBlob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import wikipedia
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler

# Check for GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f'Using device: {device}, GPU count: {num_gpus}')

# Download necessary NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('words')

# Load spaCy model
spacy_model = spacy.load('en_core_web_sm')

# Set up model and tokenizer
model_name = "/mnt/beegfs/home/jpindell2022/scratch/models/Opt-2.7b-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

if num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)
scaler = GradScaler()

def extract_segments(text, max_tokens=20, min_tokens=10):
    words = nltk.word_tokenize(text)
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens//2) if len(words[i:i+max_tokens]) >= min_tokens]

def load_gutenberg_texts():
    from nltk.corpus import gutenberg
    return [segment for fileid in gutenberg.fileids() for segment in extract_segments(gutenberg.raw(fileid))]

def load_wikipedia_texts(num_articles=10):
    segments = []
    for _ in range(num_articles):
        try:
            text = wikipedia.page(wikipedia.random()).content
            segments.extend(extract_segments(text))
        except wikipedia.exceptions.DisambiguationError:
            continue
    return segments

def corrupt_text(text):
    def syntax_break(t):
        words = t.split()
        if len(words) > 1:
            idx = random.randint(0, len(words)-1)
            words[idx] = words[idx][:len(words[idx])//2] + " " + words[idx][len(words[idx])//2:]
        return ' '.join(words)

    def semantic_drift(t):
        words = nltk.corpus.words.words()
        tokens = text.split()
        if tokens:
            idx = random.randint(0, len(tokens)-1)
            tokens[idx] = random.choice(words)
        return ' '.join(tokens)

    def tense_shift(t):
        return ' '.join([TextBlob(word).words[0].pluralize() if spacy_model(word)[0].pos_ == 'VERB' else word for word in t.split()])

    def nonsense_insert(t):
        random_words = [random.choice(nltk.corpus.words.words()) for _ in range(3)]
        return t + " " + " ".join(random_words)

    def reorder_structure(t):
        words = t.split()
        mid = len(words) // 2
        return ' '.join(words[mid:] + words[:mid])

    corruptions = [syntax_break, semantic_drift, tense_shift, nonsense_insert, reorder_structure]
    return random.choice(corruptions)(text)

class CoherenceDataset(Dataset):
    def __init__(self, coherent_texts):
        self.data = [(text, 1) for text in coherent_texts] + [(corrupt_text(text), 0) for text in coherent_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = tokenizer(text, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoded.items()}, torch.tensor(label, dtype=torch.long)

def train_model(model, dataset, epochs=3, batch_size=8):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            with autocast():
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    print("Loading datasets...")
    coherent_texts = load_gutenberg_texts() + load_wikipedia_texts(10)
    dataset = CoherenceDataset(coherent_texts)
    print("Starting training...")
    train_model(model, dataset)
    print("Saving model...")
    model.save_pretrained("coherence_model_opt2.7b")
    tokenizer.save_pretrained("coherence_model_opt2.7b")