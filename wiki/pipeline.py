'''
full pipeline
1. get reference
2. get perturbed reference base on `op_name`
3. run metrics on reference / perturbed reference

supports:
- multiple seeds
- multiple op
- multiple metrics (GPT-PPL, MLM-PPL, MAUVE-gpt2, MAUVE-roberta)
'''

import argparse
import copy
import pickle
import json
import os
from typing import List
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
from entity_mod import get_all_sents, warp_text, aggregate_stats
from metrics import get_metrics
from collections import defaultdict
from datasets import load_from_disk
from sanity_transform import batch_sanity_transform
from data.utils import postprocess_text

'''reference'''

def get_prefix(tokenizer, dataset, prefix_length, return_start_idx=False, full=False):
    success = False
    while not success:
        start_idx = random.randrange(len(dataset))
        text = ''
        for i in range(start_idx, len(dataset)):
            text += dataset[i]['text']
            text_ids = tokenizer.encode(text)
            if len(text_ids) >= prefix_length:
                prefix = tokenizer.decode(text_ids[:prefix_length])
                success = True
                break
            if full:
                break # so that the single dataset[i]['text'] need to satify required prefix_length
    if return_start_idx:
        return prefix, start_idx
    return prefix

def get_refs(prefix_tokenizer, dataset, maxlen: int, ref_num: int):
    '''get 2 non-overlapping reference'''

    ref_list = []
    blocked_prefix_idx = set()
    for i in tqdm(range(2*ref_num)):
        ref, prefix_start_idx = get_prefix(prefix_tokenizer, dataset, maxlen, return_start_idx=True, full=True)
        while prefix_start_idx in blocked_prefix_idx:
                # resample until non-blocked start idx is selected
                ref, prefix_start_idx = get_prefix(prefix_tokenizer, dataset, maxlen, return_start_idx=True, full=True)
        blocked_prefix_idx.add(prefix_start_idx)
        # if i < ref_num:
        #     ref, prefix_start_idx = get_prefix(prefix_tokenizer, dataset, maxlen, return_start_idx=True, full=True)
        #     blocked_prefix_idx.add(prefix_start_idx)
        # else:
        #     while prefix_start_idx in blocked_prefix_idx:
        #         # resample until non-blocked start idx is selected
        #         ref, prefix_start_idx = get_prefix(prefix_tokenizer, dataset, maxlen, return_start_idx=True, full=True)
        ref_list.append(postprocess_text(ref, truncate_at_period=True, postprocess_special=True))
    refs1, refs2 = ref_list[:ref_num], ref_list[ref_num:]
    return refs1, refs2

'''END reference'''
'''noise ref'''
def op_basename(op_name):
        return '-'.join(op_name.split('-')[:2])

def get_op_and_kwargs(refs, op_name: str):
    div = 2 if any([ww in op_name for ww in ['swap', 'switch', 'shuffle']]) else 1
    if op_name.startswith('con-negate-'):
        name = 'negate'
        kwargs = {'negate_prob': float(op_name.split('-')[2])}
    elif op_name.startswith('con-switch'):
        if op_name.startswith('con-switchsent'): # switchsent, switchsentnolast
            name = 'sent_switch'
        elif op_name.startswith('con-switchsubsent'): # switchsubsent, switchsubsentnolast
            name = 'switch_subsent'
        elif op_name.startswith('con-switchverb-'):
            name = 'switch_verb'
        elif op_name.startswith('con-switchnoun-'):
            name = 'switch_noun'
        elif op_name.startswith('con-switchner-'):
            name = 'switch_ner'
        kwargs = {'sent_switch_num': int(op_name.split('-')[2])}
    elif op_name.startswith('con-replacesent'): # replacesent, replacesentnolast
        name = 'sent_replace'
        kwargs = {'sent_replace_num': int(op_name.split('-')[2])}

        refs_copy = copy.deepcopy(refs)
        random.shuffle(refs_copy)
        kwargs['all_sents'], kwargs['all_sents_len'] = get_all_sents(refs_copy)
    elif op_name.startswith('con-genericner'):
        name = 'generic_ner'
        kwargs = {}
    elif op_name.startswith('flu-'): # use tianxing's code
        name = op_name
        kwargs = {}
    else:
        raise NotImplementedError
    if 'nolast' in op_name:
        kwargs['no_last'] = True
    
    return name, kwargs, div

def expand_op_name(op_names_list):
    expanded_list = []
    for op_name in op_names_list:
        if op_name == 'con-all':
            op_names_list.extend([
                'con-negate-A',
                # 'con-switchsent-A',
                'con-switchsentnolast-A',
                'con-switchverb-A',
                'con-switchnoun-A',
                'con-switchner-A',
                # 'con-replacesent-A',
                'con-replacesentnolast-A',
                'con-genericner'
            ])
        if op_name == 'con-negate-A':
            for v in [0.5, 1]:
                expanded_list.append(f'con-negate-{v}')
        elif op_name == 'con-switchsent-A':
            for v in [1, 2, 3, 6]:
                expanded_list.append(f'con-switchsent-{v}')
        elif op_name == 'con-switchsentnolast-A':
            for v in [1, 2, 6]:
                expanded_list.append(f'con-switchsentnolast-{v}')
        elif op_name == 'con-switchsubsent-A':
            for v in [1, 2, 4, 8, 16]:
                expanded_list.append(f'con-switchsubsent-{v}')
        elif op_name == 'con-switchsubsentnolast-A':
            for v in [1, 2, 4, 8, 16]:
                expanded_list.append(f'con-switchsubsentnolast-{v}')
        elif op_name == 'con-switchverb-A':
            for v in [10, 60]:
                expanded_list.append(f'con-switchverb-{v}')
        elif op_name == 'con-switchnoun-A':
            for v in [10, 60]:
                expanded_list.append(f'con-switchnoun-{v}')
        elif op_name == 'con-switchner-A':
            for v in [6, 15, 30]:
                expanded_list.append(f'con-switchner-{v}')
        elif op_name == 'con-replacesent-A':
            for v in [1, 2, 3, 4, 5, 6]:
                expanded_list.append(f'con-replacesent-{v}')
        elif op_name == 'con-replacesentnolast-A':
            for v in [1, 2, 3, 4, 5, 6]:
                expanded_list.append(f'con-replacesentnolast-{v}')
        elif op_name == 'con-replacesentnolast-A':
            for v in [1, 2, 3, 4, 5, 6]:
                expanded_list.append(f'con-replacesentnolast-{v}')
        # fluency
        elif op_name == 'flu-truncate-A':
            for v in [0.10, 0.20, 0.30, 0.40, 0.50]:
                expanded_list.append(f'flu-truncate-{v}')
        elif op_name == 'flu-randomworddrop-A':
            for v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                expanded_list.append(f'flu-randomworddrop-{v}')
        elif op_name == 'flu-randomworddrop-A1':
            for v in [0.05, 0.10, 0.15, 0.20, 0.25]:
                expanded_list.append(f'flu-randomworddrop-{v}')
        elif op_name == 'flu-randomworddrop-A2':
            for v in [0.30, 0.35, 0.40, 0.45, 0.50]:
                expanded_list.append(f'flu-randomworddrop-{v}')
        elif op_name == 'flu-randomlocalswap-A':
            for v in [0.05, 0.15, 0.30, 0.60]:
                expanded_list.append(f'flu-randomlocalswap-{v}')
        elif op_name == 'flu-randomtokenrep-A':
            for v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                expanded_list.append(f'flu-randomtokenrep-{v}')
        elif op_name == 'flu-randomtokenrep-A1':
            for v in [0.05, 0.10, 0.15, 0.20, 0.25]:
                expanded_list.append(f'flu-randomtokenrep-{v}')
        elif op_name == 'flu-randomtokenrep-A2':
            for v in [0.30, 0.35, 0.40, 0.45, 0.50]:
                expanded_list.append(f'flu-randomtokenrep-{v}')
        elif op_name == 'flu-sentencemiddleswap-A':
            for v in [1, 2, 3]:
                expanded_list.append(f'flu-sentencemiddleswap-{v}')
        elif op_name == 'flu-lemmatizeverb-A':
            for v in [0.5, 1.0]:
                expanded_list.append(f'flu-lemmatizeverb-{v}')
        elif op_name == 'flu-removepreposition-A':
            for v in [0.4, 0.7, 1.0]:
                expanded_list.append(f'flu-removepreposition-{v}')
        elif op_name == 'flu-noisepunct-A':
            for v in [0.5, 1.0]:
                expanded_list.append(f'flu-noisepunct-{v}')
        elif op_name == 'flu-removestopwords-A':
            for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                expanded_list.append(f'flu-removestopwords-{v}')
        elif op_name == 'flu-removearticle-A':
            for v in [0.5, 1.0]:
                expanded_list.append(f'flu-removearticle-{v}')
        
        elif op_name == 'flu-all':
            # adapted from /data1/groups/txml/projects/metricnlg_2205/BARTScore/SUM/score.py
            ht = ('flu-truncate,flu-randomworddrop,flu-randomlocalswap,flu-randomtokenrep,flu-sentencemiddleswap,flu-lemmatizeverb,flu-removepreposition,' + 'flu-noisepunct,flu-removestopwords,flu-removearticle,')
            ht = ht.replace('flu-truncate,', ''.join(['flu-truncate-A,'.replace('A', str(p)) for p in [0.10, 0.20, 0.30, 0.40, 0.50]])) # old: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] 
            ht = ht.replace('flu-randomworddrop,', ''.join(['flu-randomworddrop-A[seed],'.replace('A', str(p)) for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]))
            ht = ht.replace('flu-randomlocalswap,', ''.join(['flu-randomlocalswap-A[seed],'.replace('A', str(p)) for p in [0.05, 0.15, 0.30, 0.60]]))
            ht = ht.replace('flu-randomtokenrep,', ''.join(['flu-randomtokenrep-A[seed],'.replace('A', str(p)) for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]))
            ht = ht.replace('flu-sentencemiddleswap,', ''.join(['flu-sentencemiddleswap-A[seed],'.replace('A', str(p)) for p in [1, 2, 3]]))  
            ht = ht.replace('flu-lemmatizeverb,', ''.join(['flu-lemmatizeverb-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   
            ht = ht.replace('flu-removepreposition,', ''.join(['flu-removepreposition-A[seed],'.replace('A', str(p)) for p in [0.4, 0.7, 1.0]]))   
            ht = ht.replace('flu-noisepunct,', ''.join(['flu-noisepunct-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   
            ht = ht.replace('flu-removestopwords,', ''.join(['flu-removestopwords-A[seed],'.replace('A', str(p)) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))   
            ht = ht.replace('flu-removearticle,', ''.join(['flu-removearticle-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   
            ht = ht.replace('[seed]', '')
            assert(ht.endswith(',')); ht = ht[:-1]
            expanded_list.extend(ht.split(','))
        else:
            if not ('all' in op_name):
                expanded_list.append(op_name)
    return expanded_list

def get_noised_ref(refs: List[str], op_name: str) -> List[str]:
    '''
    get noised ref and compute editratio
    '''
    if op_name == 'ref':
        return refs, 0
    op, kwargs, div = get_op_and_kwargs(refs, op_name)
    if op.startswith('flu-'):
        args = argparse.Namespace(debug_transform=False)
        transform_d = {'src': ['STUB' for _ in range(len(refs))]} # TODO: change this FAKE alternative reference
        warped_texts, stat_d = batch_sanity_transform(args, refs, op, transform_d, seed=None)
        edit_ratio = stat_d['edit_ratio']
    else:
        warped_texts, stats_list = [], []
        for text in tqdm(refs):
            warped_text, stats = warp_text(text, op, **kwargs)
            warped_texts.append(warped_text), stats_list.append(stats)
        stats_dict, _ = aggregate_stats(stats_list)
        edit_ratio = stats_dict['t_norm_ed'] / div
    return warped_texts, edit_ratio

'''END noise ref'''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to wiki 103 train dir')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--ref_num', type=int, default=1000)
    parser.add_argument('--maxlen', type=int, default=256)

    parser.add_argument('--metrics', type=str, nargs='+', required=True)
    parser.add_argument('--op_names', type=str, nargs='+', required=True)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')

    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    metrics_list = get_metrics(args.metrics)
    seeds = [int(sd) for sd in args.seeds.split(',')]
    dataset = load_from_disk(args.data_dir)

    PREFIX_TOKENIZER_STR = 'gpt2'
    prefix_tokenizer = AutoTokenizer.from_pretrained(PREFIX_TOKENIZER_STR)

    expanded_op_names = expand_op_name(args.op_names)
    print(f'===\nExpanded op_names: {expanded_op_names}')

    result_d = {metric.name: {k:defaultdict(list) for k in expanded_op_names} for metric in metrics_list}
    for seed in seeds:
        print(f'===\nseed:{seed}')
        set_seed(seed)
        # check cache
        cache_path = os.path.join(args.output_dir, 'cache', f'{os.path.basename(args.data_dir)}_{args.ref_num}_{args.maxlen}', f'seed{seed}', 'refs.pkl')
        if os.path.exists(cache_path) and not args.no_cache:
            with open(cache_path, 'rb') as f:
                refs1, refs2 = pickle.load(f)
        else:
            refs1, refs2 = get_refs(prefix_tokenizer, dataset, args.maxlen, args.ref_num)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump((refs1, refs2), f)
        ###
        for metric in metrics_list:
            print(f'===\nmetric:{metric.name}')
            for op_name in expanded_op_names:
                print(f'===\nop:{op_name}')
                output_path = os.path.join(args.output_dir, metric.name, f'{op_basename(op_name)}.json')
                if os.path.exists(output_path) and not args.force:
                    print('score cached! continuing...')
                    continue
                # check cache
                cache_path = os.path.join(args.output_dir, 'cache', f'{os.path.basename(args.data_dir)}_{args.ref_num}_{args.maxlen}', f'seed{seed}', f'noise-{op_name}.pkl')
                if os.path.exists(cache_path) and not args.no_cache:
                    with open(cache_path, 'rb') as f:
                        noised, edit_ratio = pickle.load(f)
                else:
                    set_seed(seed)
                    noised, edit_ratio = get_noised_ref(refs2, op_name)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump((noised, edit_ratio), f)
                ###
                # check cache
                cache_path = os.path.join(args.output_dir, 'cache', f'{os.path.basename(args.data_dir)}_{args.ref_num}_{args.maxlen}', f'seed{seed}', f'noise-{op_name}_metric-{metric.name}.pkl')
                if os.path.exists(cache_path) and not args.no_cache:
                    with open(cache_path, 'rb') as f:
                        score = pickle.load(f)
                else:
                    score = metric.score(noised, refs1)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(score, f)
                ###
                # score = metric.score(noised, refs1)
                result_d[metric.name][op_name]['score'].append(score)
                result_d[metric.name][op_name]['edit_ratio'].append(edit_ratio)
    print('Aggregating scores...')
    for metric_name, metric_d in result_d.items():
        final_result_d = defaultdict(dict)
        for op_name, d in metric_d.items():
            output_path = os.path.join(args.output_dir, metric_name, f'{op_basename(op_name)}.json')
            if os.path.exists(output_path) and not args.force:
                print('score cached! continuing...')
                continue
            agg_result = {
                'mean': float(np.array(d['score']).mean()),
                'std': float(np.array(d['score']).std()),
                'edit_ratio': float(np.array(d['edit_ratio']).mean()),
            }
            final_result_d[op_basename(op_name)][op_name] = agg_result
        
        for op_base, res_d in final_result_d.items():
            output_path = os.path.join(args.output_dir, metric_name, f'{op_base}.json')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(res_d, f)
