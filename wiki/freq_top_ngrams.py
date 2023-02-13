import argparse
from util import load_file_by_line, write_file_by_line
from transformers import AutoTokenizer
import os
import random
from string import punctuation
import pickle
from tqdm import tqdm

def get_output_path(args):
    tokenizer_str = f'_tok{args.tokenizer}' if args.tokenizer is not None else ''
    output_path = f'data/top_ngram{tokenizer_str}/n{args.num}_max{args.max_length}/{args.gram}gram_topk{args.topk}.txt'
    assert not os.path.exists(output_path)
    return output_path

def get_top_token_rep_text(common_ngram, tok_func, gram, topk, maxlen, ignore_punct=True):
    topk_ngram = []
    for ngram, frq in common_ngram:
        if len(ngram) == 0: continue
        if ignore_punct and any(p in ngram for p in punctuation):
            # only works for not tokenized ids
            continue
        topk_ngram.append(ngram)
        if len(topk_ngram) == topk:
            break
    decoded_ngram = [tok_func(ngram) for ngram in topk_ngram]

    text = ''
    len_t = 0
    while True:
        to_add = random.choice(decoded_ngram)
        if len_t + gram <= maxlen:
            text += f' {to_add}'
            len_t += gram
        else:
            break
    return text
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('-t', '--tokenizer', type=str, default=None, help='tokenizer for huggingface AutoModelForCausalLM to base the freqs on')
    parser.add_argument('-c', '--counter_path', type=str, required=True)

    parser.add_argument('-g', '--gram', type=int, default=1)
    parser.add_argument('--topk', type=int, default=10, help='random from top k ngrams')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    output_path = get_output_path(args)
    random.seed(args.seed)

    with open(args.counter_path, 'rb') as f:
        freqs = pickle.load(f)

    if args.tokenizer is not None:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok_func = lambda x: tok.decode([int(el) for el in " ".split(x)]) if isinstance(x, str) else tok.decode(x)
    else:
        tok_func = lambda x: x
    
    texts = []
    common_ngram = freqs.most_common(args.topk*10) # save time
    for i in tqdm(range(args.num)):
        text = get_top_token_rep_text(common_ngram, tok_func, args.gram, args.topk, args.max_length)
        texts.append(text)

    write_file_by_line(output_path, texts)