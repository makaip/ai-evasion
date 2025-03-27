import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load GPT-2 model for Perplexity calculation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def load_lines(filename):
    """ Load text from a file, treating each line as a separate sample """
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

# Load text files (one sample per line)
text_human = load_lines("human.txt")
text_ai = load_lines("ai.txt")
text_model1 = load_lines("branchingalg.txt")
text_model2 = load_lines("iterativesampling.txt")

# Function to compute Perplexity per line
def calculate_perplexity(text_samples):
    """ Calculate the perplexity for each line and return a list of scores """
    perplexities = []
    for text in text_samples:
        encodings = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
        perplexities.append(torch.exp(loss).item())  # Perplexity = e^(loss)
    return perplexities

# Compute Perplexity Scores
perp_human = calculate_perplexity(text_human)
perp_ai = calculate_perplexity(text_ai)
perp_model1 = calculate_perplexity(text_model1)
perp_model2 = calculate_perplexity(text_model2)

# Two-Sample t-Test (Human vs. Model Outputs)
t_stat1, p_val1 = ttest_ind(perp_human, perp_model1, equal_var=False)
t_stat2, p_val2 = ttest_ind(perp_human, perp_model2, equal_var=False)

print("\nTwo-Sample t-Test Results:")
print(f"Human vs. Branching Algorithm: t={t_stat1:.4f}, p={p_val1:.4f}")
print(f"Human vs. Iterative Sampling: t={t_stat2:.4f}, p={p_val2:.4f}")

# ROC & AUC Analysis (Using Real Samples)
# Assign labels: 0 = Human, 1 = AI-generated
y_true = [0] * len(perp_human) + [1] * len(perp_ai)
y_scores = perp_human + perp_ai  # Assuming Perplexity as detection score (higher = more human-like)

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for AI Detector")
plt.legend(loc="lower right")
plt.show()

print(f"\nROC AUC Score: {roc_auc:.4f}")
