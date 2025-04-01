import os
import pickle
import nltk
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import spacy
import random
from textblob import TextBlob
from nltk.corpus import gutenberg
from transformers import AutoTokenizer

# Setup logging
import logging
log_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/preprocess.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Download required NLTK data (only needs to run once)
# nltk.download('gutenberg')
# nltk.download('punkt')
# nltk.download('words')

spacy_model = spacy.load('en_core_web_sm')

model_name = "/mnt/beegfs/home/jpindell2022/scratch/models/Opt-2.7b-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

DATASET_PATH = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/cached_dataset.pkl"

def extract_segments(text, max_tokens=512, min_tokens=256):
    words = nltk.word_tokenize(text)
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens//2) if len(words[i:i+max_tokens]) >= min_tokens]

def load_gutenberg_texts():
    logging.info("Loading Gutenberg texts...")
    return [segment for fileid in gutenberg.fileids() for segment in extract_segments(gutenberg.raw(fileid))]

def load_wikipedia_texts(num_articles=50):
    logging.info(f"Fetching {num_articles} Wikipedia articles...")
    segments = []
    for _ in range(num_articles):
        try:
            text = wikipedia.page(wikipedia.random()).content
            segments.extend(extract_segments(text))
        except (DisambiguationError, PageError):
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
        tokens = t.split()
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

class CoherenceDataset(torch.utils.data.Dataset):
    def __init__(self, coherent_texts):
        logging.info("Preprocessing and caching dataset...")
        self.data = []

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

        logging.info(f"Created dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    logging.info("Starting dataset preprocessing...")

    coherent_texts = load_gutenberg_texts() + load_wikipedia_texts(50)
    dataset = CoherenceDataset(coherent_texts)

    logging.info(f"Saving dataset to {DATASET_PATH}")
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)

    logging.info("Dataset preprocessing completed.")
