# datasets/dialogue/download.py
from datasets import load_dataset
import os

os.makedirs("datasets/dialogue", exist_ok=True)

ds = load_dataset("OpenAssistant/oasst1", split="train")

with open("datasets/dialogue/dialogue.txt", "w", encoding="utf-8") as f:
    for item in ds:
        text = item["text"]
        if text and len(text) > 50:
            f.write(text.replace("\n", " ") + "\n")
