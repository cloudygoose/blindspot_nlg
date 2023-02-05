import os, sys

def clean_zh(sen):
    if sen.endswith('？') or sen.endswith('。') or sen.endswith('！'):
        return sen
    else:
        return sen + '。'

def load_select(sp):
    en_lines = [sen.strip() for sen in open('ted_{}_en-zh.raw.en'.format(sp)).readlines()]
    zh_lines = [clean_zh(sen.strip()) for sen in open('ted_{}_en-zh.raw.zh'.format(sp)).readlines()]
    
    ld, co = [], 0
    for idl in open('select_id_{}.txt'.format(sp)).readlines():
        idl = idl.strip()
        if len(idl) < 2: continue
        st, ed = idl.split('-')
        st, ed = int(st) - 1, int(ed)
        assert(ed > st + 2)
        zh_l, en_l = ' '.join(zh_lines[st:ed]), ' '.join(en_lines[st:ed])
        co += 1
        ld.append({'zh': zh_l, 'en0': en_l, 'en1': '', 'idl': idl})
    
    print(co, 'loaded from', sp)
    return ld

ld1 = load_select('test')
ld2 = load_select('dev')
ld = ld1 + ld2

for rid in [0, 1]:
    idx, fn = 0, 'translate/zh_en{}.txt'.format(rid)
    print('output to', fn)
    fout = open(fn, 'w')
    for ss in ld:
        zh_l = ss['zh']
        en_l = ss['en{}'.format(rid)]
        fout.write('{} {} ZH: {}\n'.format(idx, ss['idl'], zh_l))
        fout.write('{} {} EN: {}\n'.format(idx, ss['idl'], en_l))
        idx += 1
    fout.close() 