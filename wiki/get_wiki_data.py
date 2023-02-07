from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
dataset.save_to_disk("data/wikitext-103-raw")