import random
import json
import torch
import ollama

# Load your secondary model (trimming model)
# secondary_model = torch.load("path/to/model.pth")
# secondary_model.eval()

def score_branch(branch_text):
    """
    Score a branch for coherence using the secondary model.
    Replace this with your model's inference logic.
    """
    return random.random()

def generate_next_tokens(prefix, step_size=2, top_k=4, top_p=0.9):
    """
    Generate multiple token continuations from the prefix using Ollama.
    """
    response = ollama.generate(
        model="llama3",
        prompt=prefix,
        options={"num_predict": step_size, "top_k": top_k, "top_p": top_p}
    )
    tokens = response["response"].strip()
    return tokens

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
            tree[branch] = candidates
            next_level.extend(candidates)
        current_level = next_level
    
    with open("generation.json", "w") as json_file:
        json.dump(tree, json_file, indent=4)
    
    return tree

def trim_tree(tree):
    """
    Evaluate branches using the secondary model and select one based on scores.
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
    Generates a branching tree, trims it, and continues generation from the selected branch.
    """
    tree = generate_tree(prompt, tree_size, step_size, top_k, top_p)
    selected_root = trim_tree(tree)
    
    current_text = selected_root
    for _ in range(n_steps // step_size):
        tokens = generate_next_tokens(current_text, step_size=step_size, top_k=top_k, top_p=top_p)
        current_text += " " + tokens
    
    with open("generation.txt", "w") as text_file:
        text_file.write(current_text)
    
    return current_text

if __name__ == "__main__":
    prompt = "Once"
    final_text = branch_and_trim(prompt, tree_size=4, step_size=2, top_k=5, top_p=0.25, n_steps=6)
    print("\nFinal generated text:", final_text)
