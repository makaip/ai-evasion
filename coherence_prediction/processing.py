import nltk
import random
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import spacy
from textblob import TextBlob
from nltk.corpus import gutenberg

# nltk.download('gutenberg')
# nltk.download('punkt')
# nltk.download('words')

spacy_model = spacy.load('en_core_web_sm')

def extract_segments(text, max_tokens=512, min_tokens=256):
    words = nltk.word_tokenize(text)
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens//2) if len(words[i:i+max_tokens]) >= min_tokens]

def load_gutenberg_texts():
    return [segment for fileid in gutenberg.fileids() for segment in extract_segments(gutenberg.raw(fileid))]

def load_wikipedia_texts(num_articles=50):
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