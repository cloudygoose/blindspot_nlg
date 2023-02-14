import torch
import os
from nltk import word_tokenize, sent_tokenize
import pickle

def pad_to_maxlen(pt_list, pad_id):
    lengths = torch.LongTensor([pt.size(0) for pt in pt_list])
    max_length = lengths.max()
    attn_masks = []
    for i in range(len(pt_list)):
        if len(pt_list[i]) < max_length:
            pt_list[i] = torch.cat([pt_list[i], pad_id * torch.ones(max_length - len(pt_list[i])).long()], dim=0) # actually 0 is fine as pad since it's masked out
            attn_masks.append(torch.cat([torch.ones(lengths[i]).long(), torch.zeros(max_length - lengths[i]).long()], dim=0))
        else:
            attn_masks.append(torch.ones(max_length).long())
    return torch.stack(pt_list, dim=0), torch.stack(attn_masks, dim=0) # input_ids, attention_mask

def load_file_by_line(path):
    '''
    each line is an example of ref/gen text; ignore comment that starts with # (by the start of line)
    '''
    texts = []
    if path is None: return texts
    with open(path, 'r') as f:
        for _, line in enumerate(f.readlines()):
            line = line.strip()
            if not line.startswith('#'):
                texts.append(line)
    return texts

def write_file_by_line(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for text in texts:
            print(text.replace('\n', '\\n'), file=f) # make sure it's a single line

def path_wo_ext(path):
    return os.path.splitext(path)[0]

def break_text(text_list):
    tokens_list = []
    for text in text_list:
        sentences = sent_tokenize(text.lower())
        tokens = []
        for s in sentences: tokens += word_tokenize(s)
        tokens_list.append(tokens)
    return tokens_list

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def chunks(l, batch_size):
    return list(l[i:i + batch_size] for i in range(0, len(l), batch_size))