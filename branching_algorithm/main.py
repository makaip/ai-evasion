import random
import json
import torch
import logging  # Import the logging module.
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from vllm import LLMEngine, SamplingParams

# Initialize VLLM engine for the primary model.
primary_model_path = "/mnt/beegfs/groups/ouri_project/huggingface/Meta-Llama-3-70B-Instruct"
log_stats = False  # Set to True if logging statistics is required.
primary_engine = LLMEngine(primary_model_path, trust_remote_code=True)

# Load the secondary (trimming) model.
trimming_model_path = "/mnt/beegfs/home/jpindell2022/scratch/coherence_dataset/best_roberta_model.pt"
trimming_model = torch.load(trimming_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
trimming_model.eval()
# Load the corresponding tokenizer for the trimming model.
trimming_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Configure logging to write to a file.
log_file_path = "/mnt/beegfs/home/jpindell2022/projects/aiouri/ai-evasion/branching_algorithm/log.txt"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def score_branch(branch_text):
    """
    Score a branch for coherence using the secondary (trimming) model.
    The trimming model is assumed to be a binary classifier where the second class 
    (index 1) represents coherence. This function returns the probability of coherence.
    """
    inputs = trimming_tokenizer(branch_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Move inputs to device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = trimming_model(**inputs)
    logits = outputs.logits
    # Compute softmax probabilities; assume logits shape [1,2] for binary classification.
    prob_coherent = torch.softmax(logits, dim=-1)[0, 1].item()
    return prob_coherent

def generate_next_tokens(prefix, step_size=2, top_k=4, top_p=0.9):
    """
    Generate token continuations from the prefix using VLLM.
    """
    sampling_params = SamplingParams(
        max_tokens=step_size,
        top_k=top_k,
        top_p=top_p,
        # Add any additional sampling parameters as needed.
    )
    results = primary_engine.generate(prefix, sampling_params)
    # VLLM's result returns a list of generation results.
    generated_text = results[0].text
    # Remove the prompt from the generated text to get only new tokens.
    tokens = generated_text[len(prefix):].strip()
    return tokens

def append_to_file(filename, text):
    """Append text to a file."""
    with open(filename, "a") as file:
        file.write(text + "\n")

def generate_tree(prompt, tree_size, step_size=2, top_k=4, top_p=0.9):
    """
    Build a tree of continuations from the prompt.
    """
    tree = {prompt: []}
    current_level = [prompt]
    
    for _ in range(tree_size):
        next_level = []
        for branch in current_level:
            candidates = []
            for _ in range(top_k):
                tokens = generate_next_tokens(branch, step_size=step_size, top_k=top_k, top_p=top_p)
                candidate = branch + " " + tokens
                candidates.append(candidate)
                append_to_file("generation_log.txt", candidate)
            tree[branch] = candidates
            next_level.extend(candidates)
        current_level = next_level
    
    with open("generation.json", "w") as json_file:
        json.dump(tree, json_file, indent=4)
    
    return tree

def trim_tree(tree):
    """
    Evaluate branches using the secondary (trimming) model and select one based on coherence scores.
    """
    branch_scores = {branch: score_branch(branch) for branch in tree.keys()}
    branches = list(branch_scores.keys())
    scores = list(branch_scores.values())
    total = sum(scores)
    if total == 0:
        return random.choice(branches)
    probabilities = [score/total for score in scores]
    return random.choices(branches, weights=probabilities, k=1)[0]

def branch_and_trim(prompt, tree_size, step_size=2, top_k=4, top_p=0.9, n_steps=5):
    """
    Generates a branching tree, trims it using the secondary model, and continues generation from the selected branch.
    """
    tree = generate_tree(prompt, tree_size, step_size, top_k, top_p)
    selected_root = trim_tree(tree)
    
    current_text = selected_root
    append_to_file("generation_log.txt", "\nSelected Root: " + selected_root)
    
    for _ in range(n_steps // step_size):
        tokens = generate_next_tokens(current_text, step_size=step_size, top_k=top_k, top_p=top_p)
        current_text += " " + tokens
        append_to_file("generation_log.txt", current_text)
    
    with open("generation.txt", "w") as text_file:
        text_file.write(current_text)
    
    return current_text

if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    with open("cuda_status.txt", "w") as cuda_file:
        cuda_file.write(f"CUDA Available: {cuda_available}\n")
    logging.info(f"CUDA Available: {cuda_available}")
    
    prompt = "Once"
    logging.info(f"Starting branch_and_trim with prompt: {prompt}")
    final_text = branch_and_trim(prompt, tree_size=3, step_size=2, top_k=3, top_p=0.25, n_steps=3)
    logging.info(f"Final generated text: {final_text}")
    print("\nFinal generated text:", final_text)  # Retain print for console output if needed.
