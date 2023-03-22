import argparse
from util import load_file_by_line, write_file_by_line
from transformers import AutoTokenizer
import os
import random
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

def warp_text(tokenizer, allowed_ids, text, op, times, offset_period=False):
    text_ids = tokenizer.encode(text)
    if op == 'suffix': text_ids = text_ids[::-1] # reverse
    offset = 1 if offset_period and text_ids[0] == tokenizer.encode('.')[0] else 0
    for i in range(min(times, len(text_ids))):
        text_ids[offset+i] = random.choice(allowed_ids)
    if op == 'suffix': text_ids = text_ids[::-1] # reverse
    warped_text = tokenizer.decode(text_ids)
    return warped_text

def warp_text(tokenizer, allowed_ids, text, op:str, times, offset_period=False):
    # special: replast
    if op == 'replast':
        words = TreebankWordTokenizer().tokenize(text)
        warped_ids = words + words[-4:]*times
        warped_text = TreebankWordDetokenizer().detokenize(warped_ids)
        return warped_text
    text_ids = tokenizer.encode(text)
    ending_offset = 1 if offset_period and text_ids[0] == tokenizer.encode('.')[0] else 0
    max_effective_len = len(text_ids) - ending_offset # skip period if there is any
    if op.startswith('prefix'): # 'prefix', 'prefix-shuffle'
        start_pos = 0
    elif op.startswith('middle'):
        start_pos = max(0, max_effective_len//2-times//2) # e.g., 20//2-4//2 = 8, so position 8,9,10,11 out of 0,1,...,19 will be changed
    elif op.startswith('suffix'):
        start_pos = max(0, max_effective_len-times)
    else:
        raise NotImplementedError
    end_pos = min(start_pos+times, max_effective_len)

    if 'shuffle' in op:
        shuffle_part = text_ids[start_pos:end_pos].copy()
        random.shuffle(shuffle_part)
        for shuf_i, i in enumerate(range(start_pos, end_pos)):
            text_ids[i] = shuffle_part[shuf_i]
    else:
        for i in range(start_pos, end_pos):
            text_ids[i] = random.choice(allowed_ids)
    warped_text = tokenizer.decode(text_ids).strip()
    return warped_text
    
def get_allowed_ids(tokenizer):
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    allowed_ids = [i for i in range(len(tokenizer)) if i != eos_id]
    return allowed_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True, help='use relative path from project home dir!!')
    parser.add_argument('-t', '--tokenizer', type=str, default='gpt2', help='tokenizer for huggingface AutoModelForCausalLM that used to generate the text')
    parser.add_argument('--op', type=str, choices=['prefix', 'middle', 'suffix', 'replast'], default='prefix')
    parser.add_argument('--times', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    # options
    parser.add_argument('--offset_period', action='store_true', help='do no modify period at the end of generation')

    args = parser.parse_args()
    assert not args.generation[0].startswith('/'), 'use relative path from project home dir!!'
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    allowed_ids = get_allowed_ids(tokenizer)
    for gen in args.generation:
        texts = load_file_by_line(gen)
        warped_texts = [warp_text(tokenizer, allowed_ids, text, args.op, args.times, args.offset_period) for text in texts]
        suffix = '-leaveperiod' if args.offset_period else ''
        # new_path = os.path.join('gen_mod', f'repl-{args.op}-{args.times}{suffix}', gen[4:]) # ignore gen/ in front
        gen_name_str, gen_name_ext = os.path.splitext(gen)
        new_path = f'{gen_name_str}_repl-{args.op}-{args.times}{suffix}{gen_name_ext}'
        print(f'output path: {new_path}')
        write_file_by_line(new_path, warped_texts)


