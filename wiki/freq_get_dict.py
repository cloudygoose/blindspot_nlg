import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import os
import pickle
from nltk import word_tokenize

def ngram_list(word_list, n):
    if n == 1:
        # effect: if word_list are tokenized integer ids, they will NOT be changed to str
        return word_list
    if len(word_list) < n:
        return [" ".join(word_list)]
    return [" ".join([str(el) for el in word_list[i:i+n]]) for i in range(0, len(word_list)+1-n)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='huggingface dataset (not dict)')
    parser.add_argument('-t', '--tokenizer', type=str, default=None, help='tokenizer for huggingface AutoModelForCausalLM to base the freqs on; use ')
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-g', '--gram', type=int, default=1)
    parser.add_argument('-f', '--force', action='store_true')

    args = parser.parse_args()
    assert args.force or not os.path.exists(args.output_path), f'output path {args.output_path} exists, stopped...'
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    dataset = load_from_disk(args.dataset)
    if args.tokenizer is not None:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok_func = lambda x: tok.encode(x)
    else:
        tok_func = word_tokenize
    counts = Counter()
    for i in tqdm(range(len(dataset))):
        counts.update(ngram_list(tok_func(dataset[i]['text'].lower()), args.gram))

    with open(args.output_path, 'wb') as f:
        pickle.dump(counts, f)