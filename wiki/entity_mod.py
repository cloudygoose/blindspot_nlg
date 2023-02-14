import argparse
from collections import defaultdict
from util import load_file_by_line, write_file_by_line, load_pkl
import os
import random
from tqdm import tqdm
import re
from nltk import sent_tokenize, word_tokenize
import editdistance
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import spacy
import pyinflect
def load_spacy():
    # spacy.require_gpu() # need the cupy package: conda install -c conda-forge cupy
    # return spacy.load("en_core_web_trf") # trf => transformers (roberta) https://spacy.io/usage/facts-figures
    return spacy.load("en_core_web_sm")

# global vars and constants
NLP = None
INITIALIZED_PERTERB = False

# CARDINAL Numerals that do not fall under another type
# DATE Absolute or relative dates or periods
# EVENT Named hurricanes, battles, wars, sports events, etc.
# FAC Buildings, airports, highways, bridges, etc.
# GPE Countries, cities, states
# LANGUAGE Any named language
# LAW Named documents made into laws.
# LOC Non-GPE locations, mountain ranges, bodies of water
# MONEY Monetary values, including unit
# NORP Nationalities or religious or political groups
# ORDINAL "first", "second", etc.
# ORG Companies, agencies, institutions, etc.
# PERCENT Percentage, including "%"
# PERSON People, including fictional
# PRODUCT Objects, vehicles, foods, etc. (not services)
# QUANTITY Measurements, as of weight or distance
# TIME Times smaller than a day
# WORK_OF_ART Titles of books, songs, etc.

EOS_PUNCT = ['.', '!', '?']
ARTICLES = ['some', 'a', 'an', 'the']
MAX_ARTICLE_LENGTH=4

# replace NER with generic names
NER_TRANSFORM = {
    #'DATE': (['some', 'a'], 'date'),
    #'EVENT': (['some', 'an'], 'event'),
    'FAC': (['some', 'a'], 'place'),
    'GPE': (['some', 'a'], 'place'),
    #'LANGUAGE': (['some', 'a'], 'language'),
    'LOC': (['some', 'a'], 'place'),
    #'MONEY': (['some'], 'money'),
    'ORG': (['some', 'an'], 'organization'),
    'PERSON': (['some', 'a'], 'person'),
    'PRODUCT': (['some', 'a'], 'product'),
    #'WORK_OF_ART': (['some', 'a'], 'work of art')
}

# # replace NER with high freq NER of that type
# NER_FIX_TRANSFORM = {
#     'CARDINAL': 'one',
#     'DATE': 'today',
#     'ORDINAL': 'first',
# }

def generic_ner(text, **kwargs):
    '''
    replace NER of a particular type with generic names
    '''
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)
    s_prob = kwargs.get('prob', 1)

    new_text = text
    ner_types = set()
    for e in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        if not e.label_ in NER_TRANSFORM.keys(): continue
        if not (random.random() < s_prob): continue
        new_text = new_text[:e.start_char] + e.label_ + new_text[e.end_char:]
        ner_types.add(e.label_)

    tt = new_text.split(); new_tt = []
    for i, ww in enumerate(tt):
        if not (ww.lower() in ['a', 'an', 'the'] and i < len(tt) - 1 and tt[i + 1] in ner_types): new_tt.append(ww)
    new_text = ' '.join(new_tt)

    for ner_type in ner_types:
        if not ner_type in NER_TRANSFORM.keys(): continue
        start_idx = new_text.find(ner_type)
        while start_idx != -1:
            new_ent_entry = ''
            pref, pref_all = new_text[max(0, start_idx-MAX_ARTICLE_LENGTH-2):start_idx], new_text[:start_idx]
            allowed_articles, entry = NER_TRANSFORM[ner_type]
            nlp_pref_all = NLP(pref_all)
            if (not any(f' {art} ' in pref for art in ARTICLES)) and (not (len(nlp_pref_all) >= 1 and nlp_pref_all[-1].pos_ in ['ADJ', 'NOUN'])): # f' {art} ' to parse full word
                new_ent_entry += random.choice(allowed_articles) + ' '
            new_ent_entry += entry
            if any(f'{p} ' in pref[-3:] for p in EOS_PUNCT + ['\\n']):
                # start of sentence
                new_ent_entry = new_ent_entry.capitalize()
            new_text = new_text[:start_idx] + new_ent_entry + new_text[start_idx+len(ner_type):]

            start_idx = new_text.find(ner_type)

    # __import__('pdb').set_trace()
    return new_text

def swap_ner(text, **kwargs):
    '''
    swap ner_type_a instances and ner_type_b instances
    '''
    ner_type_a, ner_type_b = kwargs.get('ner_type_a'), kwargs.get('ner_type_b')
    assert ner_type_a is not None and ner_type_b is not None
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)

    new_text = text
    ner_type_to_inst = defaultdict(list)
    # first pass: get all relevant NER instances
    for e in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        ner_type = e.label_
        if not ner_type in [ner_type_a, ner_type_b]: continue
        ner_inst = new_text[e.start_char:e.end_char]
        ner_type_to_inst[ner_type].append(ner_inst)

    # if does not contain both types we want to swap, leave text as original
    if not (ner_type_a in ner_type_to_inst and ner_type_b in ner_type_to_inst):
        return new_text
    
    # second pass: make actual ner swap
    for e in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        ner_type = e.label_
        if not ner_type in [ner_type_a, ner_type_b]: continue
        ner_other_type = ner_type_a if ner_type != ner_type_a else ner_type_b
        new_ner_inst = random.choice(ner_type_to_inst[ner_other_type])
        new_text = new_text[:e.start_char] + new_ner_inst + new_text[e.end_char:]

    return new_text

def switch_verb(text, **kwargs):
    num = kwargs.get('sent_switch_num')
    assert isinstance(num, int)
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)

    tokens = []
    to_switch = []
    for i, t in enumerate(doc):
        tokens.append(t.text_with_ws)
        if t.pos_ == 'VERB':
            to_switch.append((i, t.tag_)) # .tag_ detailed type
    
    success = 0
    if len(to_switch) <= 2:
        # to avoid infinite loop
        # sents = sents[::-1]
        # TODO: reverse or unchanged? for now just unchanged
        pass
    else:
        last_i, last_j = -1, -1
        it = 0
        while success < num:
            it += 1
            # print(f'it={it}')
            if it > 100:
                print('too much trials...')
                print(f'success wants: {num}, got: {success}')
                print('skipping...')
                break
            i, j = random.sample(range(len(to_switch)), k=2)
            if not (((last_i == i) and (last_j == j)) or ((last_i == j) and (last_j == i))):
                ### Actual switch
                i_pos, i_type = to_switch[i]
                j_pos, j_type = to_switch[j]
                i_token = tokens[i_pos].strip()
                i_token_j_infl = NLP(i_token)[0]._.inflect(j_type)
                if i_token_j_infl is None:
                    print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... skipping')
                    continue
                    # print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... using original form instead')
                    # i_token_j_infl = i_token # failure protection
                j_token = tokens[j_pos].strip()
                j_token_i_infl = NLP(j_token)[0]._.inflect(i_type)
                if j_token_i_infl is None:
                    print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... skipping')
                    continue
                    # print(f'warning: failed to inflect [{j_token}]({j_type}) to {i_type}... using original form instead')
                    # j_token_i_infl = j_token # failure protection
                tokens[i_pos] = tokens[i_pos].replace(i_token, j_token_i_infl)
                tokens[j_pos] = tokens[j_pos].replace(j_token, i_token_j_infl)
                ###
                last_i, last_j = i, j
                success += 1
    new_text = ''.join(tokens)
    return new_text

def switch_noun(text, **kwargs):
    num = kwargs.get('sent_switch_num')
    assert isinstance(num, int)
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)

    tokens = []
    to_switch = []
    for i, t in enumerate(doc):
        tokens.append(t.text_with_ws)
        if t.pos_ == 'NOUN':
            to_switch.append((i, t.tag_)) # .tag_ detailed type
    
    success = 0
    if len(to_switch) <= 2:
        # to avoid infinite loop
        # sents = sents[::-1]
        # TODO: reverse or unchanged? for now just unchanged
        pass
    else:
        last_i, last_j = -1, -1
        while success < num:
            i, j = random.sample(range(len(to_switch)), k=2)
            if not (((last_i == i) and (last_j == j)) or ((last_i == j) and (last_j == i))):
                ### Actual switch
                i_pos, i_type = to_switch[i]
                j_pos, j_type = to_switch[j]
                i_token = tokens[i_pos].strip()
                j_token = tokens[j_pos].strip()
                tokens[i_pos] = tokens[i_pos].replace(i_token, j_token)
                tokens[j_pos] = tokens[j_pos].replace(j_token, i_token)
                ###
                last_i, last_j = i, j
                success += 1
    new_text = ''.join(tokens)
    return new_text

def shuffle_ner(text, **kwargs):
    '''
    random shuffle all NERs in a text paragraph
    '''
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)
    ents_list = list(doc.ents)
    random.shuffle(ents_list)

    new_text = text
    for i, e in enumerate(reversed(doc.ents)): # reversed to not modify the offsets of other entities when substituting
        new_ner_str = ents_list[i].text
        new_text = new_text[:e.start_char] + new_ner_str + new_text[e.end_char:]

    return new_text

def switch_ner(text, **kwargs):
    '''
    random shuffle all NERs in a text paragraph
    '''
    num = kwargs.get('sent_switch_num')
    assert isinstance(num, int)
    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)
    if len(doc.ents) == 0:
        return text

    non_ents_segs = []
    ents_segs = []
    # [non_ent][ent][non_ent]...[non_ent][ent][non_ent]
    last_e = None
    for e in doc.ents:
        if last_e is None:
            # i == 0
            non_ents_segs.append(text[:e.start_char])
        else:
            non_ents_segs.append(text[last_e.end_char:e.start_char])
        last_e = e
        ents_segs.append(text[e.start_char:e.end_char])
    non_ents_segs.append(text[last_e.end_char:])

    to_switch = ents_segs
    success = 0
    if len(to_switch) <= 2:
        # to avoid infinite loop
        # sents = sents[::-1]
        # TODO: reverse or unchanged? for now just unchanged
        pass
    else:
        last_i, last_j = -1, -1
        while success < num:
            i, j = random.sample(range(len(to_switch)), k=2)
            if not (((last_i == i) and (last_j == j)) or ((last_i == j) and (last_j == i))):
                ### Actual switch
                to_switch[i], to_switch[j] = to_switch[j], to_switch[i]
                ###
                last_i, last_j = i, j
                success += 1
    new_text = ''
    for i in range(len(ents_segs)):
        new_text += non_ents_segs[i]
        new_text += ents_segs[i]
    new_text += non_ents_segs[-1]

    return new_text

TOPFREQ_CACHE = {}
def topfreq_ner(text, **kwargs):
    '''
    replace all NERs with one of the `topfreq_topk` most frequent ocurrences of their type with rate `topfreq_prob`
    '''
    topk = kwargs.get('topfreq_topk')
    prob = kwargs.get('topfreq_prob')
    ner_freqs = kwargs.get('ner_freqs')
    assert ner_freqs is not None

    global NLP
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(text)
    global TOPFREQ_CACHE

    new_text = text
    for e in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        if random.random() >= prob: continue
        ner_type = e.label_
        topk_common = TOPFREQ_CACHE.get(ner_type)
        if topk_common is None:
            # compute and cache
            topk_common = ner_freqs[ner_type].most_common(topk) # list (name, count) tuple
            TOPFREQ_CACHE[ner_type] = topk_common
        chosen_ner_name = random.choice(topk_common)[0]
        new_text = new_text[:e.start_char] + chosen_ner_name + new_text[e.end_char:]

    return new_text

def negate(text, **kwargs):
    '''
    negate sentences (did -> didn't, etc.)
    '''
    # IMPORTANT: about StopIteration runtime error: https://github.com/RaRe-Technologies/gensim/issues/2438#issuecomment-644753776
    # need to comment out ...(conda dir).../site-packages/pattern/text/__init__.py line 609

    # doc = NLP(text)
    from checklist.perturb import Perturb # thrown wordnet error when offline
    global NLP
    if NLP is None:
        NLP = load_spacy()
    global INITIALIZED_PERTERB
    if not INITIALIZED_PERTERB:
        _ = Perturb.perturb([NLP('stub sentence')], Perturb.add_negation)
        INITIALIZED_PERTERB = True
    neg_prob = kwargs.get('negate_prob', 1.0)

    sents_new = []
    for sent in sent_tokenize(text):
        ret = Perturb.perturb([NLP(sent)], Perturb.add_negation)
            #' '.join([neg for pos,neg in ret['data']])
        if len(ret['data']) == 0:
            import transform_utils
            sent_negate = transform_utils.simple_negate(sent) #failed! fall back to naive negation heursitic...
            if sent_negate == sent: #still failed! just use the original one
                print(f'warning: sent_negate failed, using original sentence: {sent}')
        else:
            sent_negate = ret['data'][0][1]
        sent_negate = sent_negate.replace(' .', '.').replace('  ', ' ').strip() #sometimes it would add a space before the period
        selected_sent = sent_negate if random.random() < neg_prob else sent
        sents_new.append(selected_sent)
    new_text = ' '.join(sents_new)
    
    return new_text

def sent_switch(text, **kwargs):
    '''
    randomly pick two sentences, switch their position, and do this operation `sent_switch_num` times
    '''
    num = kwargs.get('sent_switch_num', 1)
    no_last = kwargs.get('no_last', False)
    sents = sent_tokenize(text)
    success = 0
    if len(sents) <= 2:
        # to avoid infinite loop
        sents = sents[::-1]
    elif no_last and len(sents) == 3:
        # can't perturb last sentence..
        sents = [sents[1], sents[0], sents[2]]
    else:
        last_i, last_j = -1, -1
        it = 0
        while success < num:
            it += 1
            if it == 100:
                print('infinite loop... break!')
                break
            i, j = random.sample(range(len(sents)), k=2)
            if no_last and ((i == len(sents)-1) or (j == len(sents)-1)):
                # do not replace last sentence!
                continue
            if not (((last_i == i) and (last_j == j)) or ((last_i == j) and (last_j == i))):
                sents[i], sents[j] = sents[j], sents[i]
                last_i, last_j = i, j
                success += 1
    new_text = ' '.join(sents)

    return new_text


def split_by_multiple_keep_delim(text: str, delims: list):
    '''return stripped components (subsentences)'''
    res = [text]
    for delim in delims:
        new_res = []
        for s in res:
            orig_split = s.split(delim)
            component = [c + (delim if i != len(orig_split)-1 else '') for i, c in enumerate(orig_split)]
            new_res.extend(component)
        res = new_res
    res = [s.strip() for s in res if len(s) > 0]
    return res

def switch_subsent(text, **kwargs):
    '''
    randomly pick two SUB sentences (split by "." or ","), switch their position, and do this operation `sent_switch_num` times
    '''
    num = kwargs.get('sent_switch_num', 1)
    no_last = kwargs.get('no_last', False)
    sents = split_by_multiple_keep_delim(text, [',', '.', '?', '!'])
    success = 0
    if len(sents) <= 2:
        # to avoid infinite loop
        sents = sents[::-1]
    elif no_last and len(sents) == 3:
        # can't perturb last sentence..
        sents = [sents[1], sents[0], sents[2]]
    else:
        last_i, last_j = -1, -1
        it = 0
        while success < num:
            it += 1
            if it == 100:
                print('infinite loop... break!')
                break
            i, j = random.sample(range(len(sents)), k=2)
            if no_last and ((i == len(sents)-1) or (j == len(sents)-1)):
                # do not replace last sentence!
                continue
            if not (((last_i == i) and (last_j == j)) or ((last_i == j) and (last_j == i))):
                sents[i], sents[j] = sents[j], sents[i]
                last_i, last_j = i, j
                success += 1
    new_text = ' '.join(sents)

    return new_text

def sent_shuffle(text, **kwargs):
    '''
    shuffle all sentences
    '''
    sents = sent_tokenize(text)
    random.shuffle(sents)
    new_text = ' '.join(sents)

    return new_text

def sent_replace(text, **kwargs):
    '''
    randomly pick one sentence, replace that with a random sentence from all generations, do this operation `sent_replace_num` times

    NOTE: after this operation, some lines might be longer than original max_length
    '''
    sents = sent_tokenize(text)
    num = kwargs.get('sent_replace_num', 1)
    all_sents = kwargs.get('all_sents')
    assert all_sents is not None
    no_last = kwargs.get('no_last', False)

    num_to_choose = min(len(sents)-int(no_last), num)
    if num_to_choose == 0: return text
    idx_to_replace = random.sample(range(len(sents)-int(no_last)), num_to_choose)
    for i in idx_to_replace:
        new_sent = random.choice(all_sents)
        sents[i] = new_sent
    new_text = ' '.join(sents)

    # for _ in range(num):
    #     i = random.choice(range(len(sents)))
    #     new_sent = random.choice(all_sents)
    #     sents[i] = new_sent
    # new_text = ' '.join(sents)

    return new_text

def sent_replace_leading(text, **kwargs):
    '''
    k = min(`sent_replace_num`, number of sentences in text)
    pick k leading sentence, replace that with k random sentences from all generations

    NOTE: after this operation, some lines might be longer than original max_length
    '''
    sents = sent_tokenize(text)
    num = kwargs.get('sent_replace_num', 1)
    all_sents = kwargs.get('all_sents')
    assert all_sents is not None
    for i in range(min(num, len(sents))):
        new_sent = random.choice(all_sents)
        sents[i] = new_sent
    new_text = ' '.join(sents)

    return new_text

def sent_replace_cont_leading(text, **kwargs):
    '''
    k = min(`sent_replace_num`, number of sentences in text-1)
    pick k leading sentence, replace that with k *continuous* random sentences from all generations

    NOTE: after this operation, some lines might be longer than original max_length
    '''
    sents = sent_tokenize(text)
    num = kwargs.get('sent_replace_num', 1)
    all_sents = kwargs.get('all_sents')
    all_sents_len = kwargs.get('all_sents_len')
    assert all_sents is not None and all_sents_len is not None

    num = min(num, len(sents)-1) # need at least one sentence unchanged
    trial = 0
    while True:
        idx_replace = random.randrange(len(all_sents_len)-1)
        if all_sents_len[idx_replace+1] - all_sents_len[idx_replace] >= num:
            break
        trial += 1
        assert trial <= 100, f'no continuous segment of length {num} exist in all sentences...'

    for i in range(num):
        new_sent = all_sents[all_sents_len[idx_replace]+i]
        sents[i] = new_sent
    new_text = ' '.join(sents)

    return new_text

def maskgen_rand(text, **kwargs):
    '''
    for each token, with `token_flip_prob` probability, (greedily) choose the token considered by the logits to be the most probable
    do this `iteration` swaps, i.e. each tokens is considered `iterations` steps
    '''
    model, tokenizer, token_flip_prob, iterations, use_heur = kwargs.get('mlm_model'), kwargs.get('mlm_tokenizer'), kwargs.get('token_flip_prob'), kwargs.get('maskgen_iter'), kwargs.get('mask_heur')

    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device) # (1, seqlen)
        probs = token_flip_prob * torch.ones_like(input_ids)
        last_mask_positions = torch.zeros_like(input_ids)
        for _ in range(iterations):
            mask_positions = torch.bernoulli(probs) == 1
            if use_heur:
                # remove contiguous
                for i in range(1, mask_positions.size(1)):
                    if mask_positions[0, i-1] and mask_positions[0, i]:
                        mask_positions[0, i] = False
                # remove last mask positions
                mask_positions = torch.logical_and(mask_positions, torch.logical_not(last_mask_positions))
            
            if mask_positions.long().sum() == 0:
                # skip this iteration, don't update last_mask_positions
                continue
            else:
                last_mask_positions = mask_positions.clone()
            # do actual maskgen
            input_ids[mask_positions] = tokenizer.mask_token_id            
            out = model(input_ids=input_ids)
            token_logits = out.logits # (1, seqlen, vocab_size)
            top_tokens = token_logits.topk(1, dim=2).indices.squeeze(2) # (1, seqlen 1) -> (1, seqlen)
            for i in range(input_ids.size(1)):
                if mask_positions[0, i]:
                    input_ids[0, i] = top_tokens[0, i]
    
    new_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return new_text

def maskgen_single(text, **kwargs):
    '''
    randomly mask a single token and re-generate it with mlm maskgen
    do this `iteration` swaps, i.e. each tokens is considered `iterations` steps
    '''
    model, tokenizer, iterations, use_heur = kwargs.get('mlm_model'), kwargs.get('mlm_tokenizer'), kwargs.get('maskgen_iter'), kwargs.get('mask_heur')

    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device) # (1, seqlen)
        last_mask_positions = torch.zeros_like(input_ids)
        for _ in range(iterations):
            mask_positions = torch.zeros_like(last_mask_positions)
            mask_idx = random.randrange(mask_positions.size(1))
            mask_positions[0, mask_idx] = True
            if use_heur:
                # remove contiguous
                # for i in range(1, mask_positions.size(1)):
                #     if mask_positions[0, i-1] and mask_positions[0, i]:
                #         mask_positions[0, i] = False
                # remove last mask positions
                mask_positions = torch.logical_and(mask_positions, torch.logical_not(last_mask_positions))
            
            if mask_positions.long().sum() == 0:
                # skip this iteration, don't update last_mask_positions
                continue
            else:
                last_mask_positions = mask_positions.clone()
            # do actual maskgen
            input_ids[mask_positions] = tokenizer.mask_token_id            
            out = model(input_ids=input_ids)
            token_logits = out.logits # (1, seqlen, vocab_size)
            top_tokens = token_logits.topk(1, dim=2).indices.squeeze(2) # (1, seqlen 1) -> (1, seqlen)
            # for i in range(input_ids.size(1)):
            #     if mask_positions[0, i]:
            input_ids[0, mask_idx] = top_tokens[0, mask_idx]
    
    new_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return new_text

def maskgen_topk(text, **kwargs):
    '''
    for each token, if its logits is not within `maskgen_topk` most probable ones, (greedily) choose the token considered by the logits to be the most probable
    do this `iteration` swaps, i.e. each tokens is considered `iterations` steps
    '''
    model, tokenizer, k, iterations = kwargs.get('mlm_model'), kwargs.get('mlm_tokenizer'), kwargs.get('maskgen_topk'), kwargs.get('maskgen_iter')

    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device) # (1, seqlen)
        for _ in range(iterations):
            out = model(input_ids=input_ids)
            token_logits = out.logits # (1, seqlen, vocab_size)
            top_tokens = token_logits.topk(1, dim=2).indices.squeeze(2) # (1, seqlen, 1) -> (1, seqlen)
            topk_cutoff_logit_values = token_logits.topk(k, dim=2).values[:, :, -1] # (1, seqlen, k) -> (1, seqlen)
            for i in range(input_ids.size(1)):
                if token_logits[0, i, input_ids[0, i]] < topk_cutoff_logit_values[0, i]:
                    input_ids[0, i] = top_tokens[0, i]
    
    new_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return new_text

def maskgen_rand_topk(text, **kwargs):
    '''
    for each token, with `token_flip_prob` probability, mask it. If its logits for original token is not within `maskgen_topk` most probable ones, then (greedily) choose the token considered by the logits to be the most probable
    do this `iteration` swaps, i.e. each tokens is considered `iterations` steps
    '''
    model, tokenizer, token_flip_prob, k, iterations = kwargs.get('mlm_model'), kwargs.get('mlm_tokenizer'), kwargs.get('token_flip_prob'), kwargs.get('maskgen_topk'), kwargs.get('maskgen_iter')

    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device) # (1, seqlen)
        probs = token_flip_prob * torch.ones_like(input_ids)
        for _ in range(iterations):
            orig_input_ids = input_ids.clone() # cache last iteration (or original) tokens
            mask_positions = torch.bernoulli(probs) == 1
            input_ids[mask_positions] = tokenizer.mask_token_id
            
            out = model(input_ids=input_ids)
            token_logits = out.logits # (1, seqlen, vocab_size)
            top_tokens = token_logits.topk(1, dim=2).indices.squeeze(2) # (1, seqlen 1) -> (1, seqlen)
            topk_cutoff_logit_values = token_logits.topk(k, dim=2).values[:, :, -1] # (1, seqlen, k) -> (1, seqlen)
            # print(topk_cutoff_logit_values[0,0], '<<==')
            hit = 0
            for i in range(input_ids.size(1)):
                # print(f'{token_logits[0, i, input_ids[0, i]]} vs {topk_cutoff_logit_values[0, i]}')
                if mask_positions[0, i] and (token_logits[0, i, input_ids[0, i]] < topk_cutoff_logit_values[0, i]):
                    input_ids[0, i] = top_tokens[0, i]
                    hit += 1
                else:
                    input_ids[0, i] = orig_input_ids[0, i]
            print(f'=>{hit/mask_positions.sum().item()}')
    
    new_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return new_text

# ===above: operations=== #

def ed_and_norm_ed(iterable, warped_iterable):
    '''support any hashable objects'''
    ed = editdistance.eval(iterable, warped_iterable)
    norm_ed = ed / len(iterable) # max(len(iterable), len(warped_iterable)) there are other ways to normalize: https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance https://www.csie.ntu.edu.tw/~b93076/Computation%20of%20Normalized%20Edit%20Distance%20and%20Applications.pdf
    return ed, norm_ed

def compute_stats(text, warped_text):
    ed, norm_ed = ed_and_norm_ed(text, warped_text)
    tokenized_text, tokenized_warped_text = word_tokenize(text), word_tokenize(warped_text)
    t_ed, t_norm_ed = ed_and_norm_ed(tokenized_text, tokenized_warped_text)
    
    stats = {
        'editdist': ed,
        'norm_editdist': norm_ed,
        'token_editdist': t_ed,
        't_norm_ed': t_norm_ed,
        'len_text': len(text),
        'len_warped_text': len(warped_text),
        'len_tokenized_text': len(tokenized_text),
        'len_tokenized_warped_text': len(tokenized_warped_text),
    }
    return stats

def aggregate_stats(stats_list):
    keys = stats_list[0].keys()
    agg_dict = {k: np.mean([stats[k] for stats in stats_list]) for k in keys}
    agg_repr = [f'#\t{k}\t{v}' for k,v in agg_dict.items()]
    return agg_dict, agg_repr

def warp_text(text, op, **kwargs):
    warp_func = globals()[op]
    warped_text = warp_func(text, **kwargs)
    stats = compute_stats(text, warped_text)
    return warped_text, stats

def op_identifier(args):
    suffix_str = ''
    if args.operation == 'negate':
        if args.negate_prob != 1.0:
            suffix_str += f'_negprob{args.negate_prob:.2f}'
    if args.operation in ['sent_switch', 'switch_verb', 'switch_noun', 'switch_ner']:
        suffix_str += f'_num{args.sent_switch_num}'
    if args.operation in ['sent_replace', 'sent_replace_leading', 'sent_replace_cont_leading']:
        suffix_str += f'_num{args.sent_replace_num}'
    if args.operation == 'swap_ner':
        suffix_str += f'_{args.ner_type_a}-{args.ner_type_b}'
    if args.operation == 'topfreq_ner':
        suffix_str += f'_topk{args.topfreq_topk}_p{args.topfreq_prob}'
    if args.operation == 'maskgen_rand':
        suffix_str += f'_{args.mlm_model}_p{args.token_flip_prob}_it{args.maskgen_iter}'
    if args.operation == 'maskgen_single':
        suffix_str += f'_{args.mlm_model}_it{args.maskgen_iter}'
    if args.operation == 'maskgen_topk':
        suffix_str += f'_{args.mlm_model}_k{args.maskgen_topk}_it{args.maskgen_iter}'
    if args.operation == 'maskgen_rand_topk':
        suffix_str += f'_{args.mlm_model}_p{args.token_flip_prob}_k{args.maskgen_topk}_it{args.maskgen_iter}'
    if args.mask_heur:
        suffix_str += '_heur'
    if args.no_last:
        suffix_str += '_nolast'
    return f'entity_{args.operation}{suffix_str}'

def get_all_sents(texts):
    all_sents = []
    all_sents_len = [0] # all_sents_len[i] is start position of ith text [0, len(sent[0]), len(sent[0])+len(sent[1]), ...] len(sent[i]) = all_sents_len[i+1] - all_sents_len[i]
    for text in texts:
        sentences = sent_tokenize(text)
        all_sents += sentences
        all_sents_len.append(len(sentences) + all_sents_len[-1])
    return all_sents, all_sents_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True, help='use relative path from project home dir!!')
    parser.add_argument('-op', '--operation', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-f', '--force', action='store_true')

    # negate
    parser.add_argument('--negate_prob', type=float, default=1.0, help='probability to perform negate on each sentence')
    # sent_switch
    parser.add_argument('--sent_switch_num', type=int, default=1, help='number of pairs to sample+switch')
    parser.add_argument('--no_last', action='store_true', help='does not change last sentence')
    # sent_replace
    parser.add_argument('--sent_replace_num', type=int, default=1, help='number of sents to replace w/ random sent')
    # swap_ner
    parser.add_argument('--ner_type_a', type=str, default='PERSON', help='first type of NER to swap')
    parser.add_argument('--ner_type_b', type=str, default='ORG', help='second type of NER to swap')
    # topfreq_ner
    parser.add_argument('--ner_freqs_path', default='metadata/nercounts_refpb1000_n100000_max256.pkl')
    parser.add_argument('--topfreq_topk', type=int, default=5, help='choose from topk freq NERs of a type to replace')
    parser.add_argument('--topfreq_prob', type=float, default=1, help='probability of doing the replacement for each entity')
    # maskgen
    parser.add_argument('--mlm_model', type=str, default='roberta-large')
    parser.add_argument('--token_flip_prob', type=float, default=0, help='should be a number strictly between 0,1')
    parser.add_argument('--maskgen_iter', type=int, default=1)
    parser.add_argument('--maskgen_topk', type=int, default=1)
    parser.add_argument('--mask_heur', action='store_true', help='add heuristics for mask position (non-contiguous, block past 1 iter)')

    args = parser.parse_args()
    assert not args.generation[0].startswith('/'), 'use relative path from project home dir!!'
    random.seed(args.seed)

    device = torch.cuda.current_device() if torch.cuda.is_available() else -1

    kwargs = {
        'negate_prob': args.negate_prob,
        'sent_switch_num': args.sent_switch_num,
        'no_last': args.no_last,
        'sent_replace_num': args.sent_replace_num,
        'ner_type_a': args.ner_type_a,
        'ner_type_b': args.ner_type_b,
        'topfreq_topk': args.topfreq_topk,
        'topfreq_prob': args.topfreq_prob,
        'token_flip_prob': args.token_flip_prob,
        'maskgen_iter': args.maskgen_iter,
        'maskgen_topk': args.maskgen_topk,
        'mask_heur': args.mask_heur,
        'device': device
    }

    for gen in args.generation:
        new_path = os.path.join('gen_mod', op_identifier(args), gen[4:]) # ignore gen/ in front
        assert not os.path.exists(new_path) or args.force, f'output path {new_path} already exist, stopped if not forcing!!'
        texts = load_file_by_line(gen)
        if args.operation in ['sent_replace', 'sent_replace_leading', 'sent_replace_cont_leading']:
            kwargs['all_sents'], kwargs['all_sents_len'] = get_all_sents(texts)
        if args.operation in ['topfreq_ner']:
            kwargs['ner_freqs'] = load_pkl(args.ner_freqs_path)
        if args.operation.startswith('maskgen'):
            mlm_model = AutoModelForMaskedLM.from_pretrained(args.mlm_model, local_files_only=True).to(kwargs['device'])
            mlm_model.eval()
            kwargs['mlm_model'] = mlm_model
            kwargs['mlm_tokenizer'] = AutoTokenizer.from_pretrained(args.mlm_model, local_files_only=True)

        warped_texts, stats_list = [], []
        for text in tqdm(texts):
            warped_text, stats = warp_text(text, args.operation, **kwargs)
            warped_texts.append(warped_text), stats_list.append(stats)
        stats_dict, stats_repr = aggregate_stats(stats_list)
        edit_ratio = stats_dict['t_norm_ed']
        print(f'edit_ratio: {edit_ratio}')
        outputs = stats_repr + warped_texts
        write_file_by_line(new_path, outputs)