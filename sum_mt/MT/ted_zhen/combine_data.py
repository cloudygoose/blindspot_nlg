import os, sys
import torch

NUM = 100

def load_fn(fn):
    print('loading from', fn)
    ld = []
    for l in open(fn, 'r').readlines():
        l = l.strip()
        if len(l) <= 4: continue
        tt = l.split()
        idx = int(tt[0])
        line = ' '.join(tt[3:])
        if idx > len(ld) - 1:
            assert(tt[2] == 'ZH:')
            ld.append({'idx': idx})
            assert(len(ld) == idx + 1)
            ld[-1]['zh'] = line
        else:
            assert(tt[2] == 'EN:')
            if '’' in line: #in the reference, we find there is sometimes chinese ’
                line = line.replace('’', "'")
            ld[-1]['en'] = line
    print(len(ld), 'loaded')
    assert(len(ld) == NUM)
    return ld

ld_ref = load_fn('translate/zh_en0.txt')
ld_tra = load_fn('translate/zh_en_jacktx.txt')

dd = {'src': [], 'ref-A': [], 'ref-B': []}
for i in range(NUM):
    dd['src'].append(ld_tra[i]['zh'])
    dd['ref-A'].append(ld_ref[i]['en'])
    dd['ref-B'].append(ld_tra[i]['en'])

save_fn = './combined_zh-en.save'
print('saving to', save_fn)
torch.save(dd, save_fn)