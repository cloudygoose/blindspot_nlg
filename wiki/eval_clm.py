'''partially adapted from nl-command'''
import argparse
import os 
from util import load_file_by_line, path_wo_ext, break_text, chunks
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import numpy as np
import natsort
from torch.nn.functional import log_softmax

device = torch.cuda.current_device() if torch.cuda.is_available() else -1


def mlm_perplexity(model, tokenizer, text, batch_size):
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device) # (1, seqlen)
        seqlen = input_ids.size(-1)
        expanded_input_ids = input_ids.expand(seqlen, seqlen).clone() # (seqlen, seqlen), clone so that input_ids will not be changed by later operations
        masked_idxs = torch.arange(seqlen).to(device)
        expanded_input_ids[torch.arange(seqlen).to(device), masked_idxs] = tokenizer.mask_token_id # mask different positions for different copies
        
        orig_token_logprobs = []
        for b_idx in chunks(torch.arange(seqlen).to(device), batch_size):
            b_masked_idxs = masked_idxs[b_idx] # (bz,)
            b_orig_ids = input_ids[0][b_idx] # (bz,)
            b_exp_input_ids = expanded_input_ids[b_idx, :]
            out = model(input_ids=b_exp_input_ids)
            token_logits = out.logits # (bz, seqlen, vocab_size)
            token_logprobs = log_softmax(token_logits, dim=-1)
            b_orig_token_logprobs = token_logprobs[torch.arange(b_idx.size(0)).to(device), b_masked_idxs, b_orig_ids] # extract position [i, b_masked_idxs[i], b_orig_ids[i]] for each i in batch (which is exactly the logprob of each masked position) (bz,)
            orig_token_logprobs.append(b_orig_token_logprobs)
        orig_token_logprobs = torch.cat(orig_token_logprobs, dim=0) # score for each position (seqlen,)
        
        nll = orig_token_logprobs[1:-1].mean() # strip eos, bos
        ppl = torch.exp(-nll)

    return ppl

def eval_mlm_perplexity(model, tokenizer, texts, batch_size=1, generation_file=None, K=500, name_suffix=''):
    ppls = []
    for i, text in enumerate(tqdm(texts, desc='mlm perplexity')):
        ppl = mlm_perplexity(model, tokenizer, text, batch_size)
        # try:
        #     ppl = mlm_perplexity(model, tokenizer, text, batch_size)
        # except KeyboardInterrupt:
        #     print('user pressed ctrl+C...')
        #     __import__('sys').exit()
        # except:
        #     print('ppl model error')
        #     print(f'text=<{text}>')
        #     print(f'index: {i}')
        #     continue
        
        ppls.append(ppl)
    ppls = torch.tensor(ppls).float()
    sent_avg_ppl = ppls.mean().item()

    if generation_file is not None:
        k = min(K, len(texts))
        # topk_ppls, topk_idx = (ppls * lengths).topk(k)
        topk_ppls, topk_idx = ppls.topk(k)
        ppl_suffix = f'_{name_suffix}.ppl'
        with open(f'{path_wo_ext(generation_file)}{ppl_suffix}', 'w') as f:
            print(f'sent_avg_ppl={sent_avg_ppl:.4f}', file=f)
            for i in range(len(topk_idx)):
                idx = topk_idx[i]
                print('-'*50, file=f)
                print(f'rank={i}', file=f)
                print(f'index={idx}', file=f)
                print(f'ppl={topk_ppls[i]}', file=f)
                print(f'text={texts[idx]}', file=f)   
    return sent_avg_ppl

def eval_perplexity(model, tokenizer, texts, generation_file=None, K=500, name_suffix=''):
    nlls = []
    ppls = []
    lengths = []
    for i, text in enumerate(tqdm(texts, desc='perplexity')):
        # important: gpt2 won't add bos <|endoftext|> when tokenize by default
        prefix = tokenizer.bos_token if 'GPT2Tokenizer' in type(tokenizer).__name__ else ''
        # prefix = ''
        input_ids = tokenizer.encode(prefix + text, return_tensors='pt', truncation=True).to(device)
        # TODO: prepend <|endoftext|> to input_ids
        target_ids = input_ids.clone()
        try:
            outputs = model(input_ids, labels=target_ids)
        except:
            print('ppl model error')
            print(f'text=<{text}>')
            print(f'input_ids=<{input_ids}>')
            print(f'index: {i}')
            continue
        
        nll = outputs[0].cpu().detach()
        ppl = torch.exp(nll)
        length = input_ids.size(1)
        nlls.append(nll)
        ppls.append(ppl)
        lengths.append(length)
    nlls = torch.tensor(nlls).float()
    ppls = torch.tensor(ppls).float()
    lengths = torch.tensor(lengths).float()
    total_length = lengths.sum()

    # token_avg_ppl = torch.exp(nlls.sum() / total_length)
    sent_avg_ppl = ppls.mean().item()
    weighted_avg_ppl = ((ppls * lengths).sum() / total_length).item()
    
    print(f'perplexity from model {name_suffix} out of {len(ppls)}:')
    # print(f'token_avg_ppl={token_avg_ppl:.4f}')
    print(f'sent_avg_ppl={sent_avg_ppl:.4f}')
    print(f'weighted_avg_ppl={weighted_avg_ppl:.4f}')

    if generation_file is not None:
        k = min(K, len(texts))
        # topk_ppls, topk_idx = (ppls * lengths).topk(k)
        topk_ppls, topk_idx = ppls.topk(k)
        ppl_suffix = f'_{name_suffix}.ppl'
        with open(f'{path_wo_ext(generation_file)}{ppl_suffix}', 'w') as f:
            # print(f'token_avg_ppl={token_avg_ppl:.4f}', file=f)
            print(f'sent_avg_ppl={sent_avg_ppl:.4f}', file=f)
            print(f'weighted_avg_ppl={weighted_avg_ppl:.4f}', file=f)
            for i in range(len(topk_idx)):
                idx = topk_idx[i]
                print('-'*50, file=f)
                print(f'rank={i}', file=f)
                print(f'index={idx}', file=f)
                print(f'ppl={topk_ppls[i]}', file=f)
                print(f'length={lengths[idx]}', file=f)
                print(f'weight={lengths[idx]/total_length}', file=f)
                print(f'text={texts[idx]}', file=f)       

    return sent_avg_ppl

def rep_ngram(sen_lis, num_gram=4):
    rep_lis = []
    for sen in sen_lis:
        uniq_ngram, all_ngram = {}, []
        for i in range(0, len(sen) - num_gram + 1):
            tt = ' '.join(sen[i:i + num_gram])
            if not tt in uniq_ngram: uniq_ngram[tt] = True
            all_ngram.append(tt)
        if len(all_ngram) == 0:
            print(f'warning: len(all_ngram) is 0!!! skipping... sample: {str(sen)}')
            continue
        rep = 1.0 - len(uniq_ngram) * 1.0 / len(all_ngram)
        rep_lis.append(rep)
    return np.mean(rep_lis)

def eval_repetition(texts, print_all=True):
    """
    texts: a list of strings, where each string is a single generation from model/reference
    """
    print('\n------evaluating repetition------')
    K = [1,2,3,4] if print_all else [4]
    for k in K:
        tokens_list = break_text(texts)
        rep = rep_ngram(tokens_list, k)
        print(f'{k}-gram rep = {rep:.4f}')
    
    return rep

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True)
    parser.add_argument('-m', '--model', type=str, required=True, help='huggingface AutoModelForCausalLM or AutoModelForMaskedLM')
    parser.add_argument('-s', '--output_suffix', type=str, default='', help='should contain model name; output name such as {filename}_{suffix}.ppl/.out')
    parser.add_argument('-mlm', '--masked_language_model', action='store_true', help='compute PSEUDO perplexity from masked language models')
    parser.add_argument('-bz', '--mlm_batch_size', type=int, default=32)
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--lim', type=int, default=None)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    save_dir = os.path.dirname(args.generation[-1])  # use directory of last generation file by default
    csv_save_path = os.path.join(save_dir, f'results_{args.output_suffix}.csv')
    if not args.force and os.path.exists(csv_save_path):
        print(f'Already have the same filename {csv_save_path}!! stopping...')
        raise AssertionError

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.masked_language_model:
        model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    else:
        pad_id = tokenizer.encode(tokenizer.eos_token)[0]
        model = AutoModelForCausalLM.from_pretrained(args.model, return_dict=True, pad_token_id=pad_id).to(device)
    model.eval()

    results = 'name\tppl\trep\n'
    for gen_path in natsort.natsorted(args.generation):
        if not os.path.exists(gen_path):
            print(f'WARNING: path {gen_path} does not exist, skipping...')
            continue
        if os.path.splitext(gen_path)[1] not in ['.txt', '.notrunc']:
            print(f'WARNING: {gen_path} is not a text file, skipping...')
            continue
        basename = os.path.splitext(os.path.basename(gen_path))[0]
        print(f'file basename: {basename}')
        gen = load_file_by_line(gen_path)
        if args.lim is not None:
            gen = gen[:args.lim]
        if args.masked_language_model:
            ppl = eval_mlm_perplexity(model, tokenizer, gen, args.mlm_batch_size, gen_path, name_suffix=args.output_suffix)
        else:
            ppl = eval_perplexity(model, tokenizer, gen, gen_path, name_suffix=args.output_suffix)
        rep = eval_repetition(gen)
        results += f'{basename}\t{ppl:.4g}\t{rep:.4g}\n'

    print(results)
    with open(csv_save_path, 'w') as f:
        print(results, file=f)
