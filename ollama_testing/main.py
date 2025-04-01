import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:70b"
PROMPT = "Once upon a time in a distant land,"
OUTPUT_FILE = "generation.txt"

# Define the request payload
data = {
    "model": MODEL_NAME,
    "prompt": PROMPT,
    "stream": False,
    "options": {"num_predict": 25}  # Generate 25 tokens
}

# Make the request to Ollama
response = requests.post(OLLAMA_URL, json=data)

if response.status_code == 200:
    result = response.json()
    generated_text = result.get("response", "No response received.")
    
    # Save output to file
    with open(OUTPUT_FILE, "w") as f:
        f.write(generated_text)
    
    print(f"Generated text saved to {OUTPUT_FILE}.")
else:
    print(f"Failed to generate text. Status code: {response.status_code}")