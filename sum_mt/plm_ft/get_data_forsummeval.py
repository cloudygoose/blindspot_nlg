import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
import json
import pickle
from nltk import word_tokenize, sent_tokenize

load_fn = 'saves/cnndailymail_raw.save'
print('loading', load_fn)
raw_datasets = torch.load(load_fn) #load_dataset('ccdv/cnn_dailymail', '3.0.0')

def prep(ss):
    #if ss[:5] == '(CNN)': ss = ss[5:] #there is "(CNN)" in the BARTScore CNNDM data, so I kept it
    ss = ss.replace('\r', ' ')
    ss = ss.replace('\n', ' ')
    ss = ss.replace(' .', '.')
    return ss

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

se_data = read_pickle('../SUM/SummEval/data.pkl')

breakpoint()
sample_d = {}
for sample in raw_datasets['test']:
    sample_d[sample['id']] = sample

for idx in se_data:
    idx_s = idx.split('-')[-1]
    assert(idx_s in sample_d)
    print(se_data[idx]['ref_summ'])

