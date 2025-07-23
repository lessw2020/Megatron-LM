import json
import os

from datasets import load_dataset

# Load env var for output
output_path = "/data/users/less/datasets/wikitext/wikitext_train.jsonl"
# Load the dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
# Step 3: Write as JSONL (each entry: {"text": <paragraph>})
print("Writing to", output_path)
with open(output_path, "w", encoding="utf-8") as f:
    for item in dataset:
        text = item.get("text", "").strip()
        if text:  # skip empty lines
            f.write(json.dumps({"text": text}) + "\n")
print(f"Saved Wikitext-103 to {output_path}")
