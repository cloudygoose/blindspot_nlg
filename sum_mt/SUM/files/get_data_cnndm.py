import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
import json, string

#load_fn = 'saves/cnndailymail_raw.save'
#print('loading', load_fn)
raw_datasets = load_dataset('ccdv/cnn_dailymail', '3.0.0')
#raw_datasets = torch.load(load_fn) #load_dataset('ccdv/cnn_dailymail', '3.0.0')

def prep(ss):
    #if ss[:5] == '(CNN)': ss = ss[5:] #there is "(CNN)" in the BARTScore CNNDM data, so I kept it
    #TODO: the summeval data is like below: so, it would better if i do word_tokenize, and add space between the comma and period.
    #Parts of Oregon , Washington and British Columbia are seeing the results of the smoke , wind and solar light combination . The reason people are seeing an intense red sunset is a result of smoke particles filtering out the shorter wavelength colors from the sunlight like greens , blues , yellows and purples , KOMO-TV said .
    ss = ss.replace('\r', ' ')
    for cc in string.ascii_uppercase:
        ss = ss.replace("'\n" + cc, "'. " + cc)
    ss = ss.replace('\n', ' ')
    ss = ss.replace(' .', '.')
    ss = ss.replace('\xa0', ' ')
    ss = ss.replace('  ', ' ')
    ss = ss.strip()
    return ss

save_dir = './cnndm_saves'
# a train.json/val.json file contains multiple examples as shown below.
#{"text": "This is the first text.", "summary": "This is the first summary."}
#{"text": "This is the second text.", "summary": "This is the second summary."}
#for split in ['train', 'validation', 'test']:
for split in ['test']:
    out_fn = save_dir + '/' + split + '.json'
    print('working on split', split)
    with open(out_fn, 'w') as fout:
        for line_num, sample in enumerate(raw_datasets[split]):
            if line_num % 2000 == 0: print(line_num)
            #if 'rap video they made set to theme' in sample['highlights'] or 'claimed antibiotics could' in sample['highlights']:
            #    breakpoint()
            ss_now = {'text': prep(sample['article']), 'summary': prep(sample['highlights']), 'id': sample['id']}
            fout.write(json.dumps(ss_now) + '\n')
            #breakpoint()

