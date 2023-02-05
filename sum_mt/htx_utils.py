import os, sys, math
import nltk
import numpy as np
import logging
import torch
import random
import time, collections
logger = logging.getLogger()

from rouge_score import rouge_scorer

def ref_rouge(refs, samples):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    assert(len(refs) == len(samples))
    r2_lis, rl_lis = [], []
    max_rl, max_rl_s = 0, None
    for ref, sample in zip(refs, samples):
        if isinstance(ref, list) and isinstance(ref[0], list) and len(ref) == 1:
            ref = ref[0]
        if isinstance(ref, list):
            res = ' '.join(ref)
        if isinstance(sample, list):
            sample = ' '.join(sample)
        
        ss = scorer.score(' '.join(ref), ' '.join(sample))
        r2_lis.append(ss['rouge2'].fmeasure)
        rl = ss['rougeL'].fmeasure
        rl_lis.append(rl)
        if rl > max_rl:
            max_rl, max_rl_s = rl, (ref, sample)
    res = {'rouge2': np.mean(r2_lis), 'rougeL': np.mean(rl_lis)}
    return res
