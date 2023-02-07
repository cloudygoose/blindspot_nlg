import re, time
import logging, random, math
import torch
import numpy as np
from nltk import word_tokenize, sent_tokenize
import spacy, editdistance
from checklist.perturb import Perturb

import transform_utils

nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger()

# global vars and constants
NLP_TRF = None
INITIALIZED_PERTERB = False

def load_spacy_trf():
    spacy.require_gpu() # need the cupy package: conda install -c conda-forge cupy
    return spacy.load("en_core_web_trf") # trf => transformers (roberta) https://spacy.io/usage/facts-figures
    #return spacy.load('en_core_web_sm')

def compute_lcs(line_a, line_b):
    ww_a, ww_b = line_a.split(), line_b.split()
    lcs_num = transform_utils.lcs(ww_a, ww_b)
    return lcs_num

def batch_sanity_transform(args, lines, mode, transform_d, seed = None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);  
    srcs = transform_d['src']
    tf_lines, stat_d = [], {}
    edit_d, ori_len, changed = [], 0, []
    div2_warn = False
    for i, line in enumerate(lines):
        tf_line, sd_now = sanity_transform(args, line, mode, src_line = srcs[i], idx = i, transform_d = transform_d)
        tf_lines.append(tf_line)
        changed.append(tf_line != line)
        ed = editdistance.eval(tf_line.split(), line.split()) * 1.0 / len(line.split());
        if any([ww in mode for ww in ['swap', 'switch', 'shuffle']]):
            if not div2_warn: 
                logger.info('===WARNING=== swap-based noise %s, divide the edit-ratio by 2...', mode); time.sleep(2);
            div2_warn = True; ed = ed / 2.0;
        edit_d.append(ed)
    
    stat_d['edit_ratio'], stat_d['change_ratio'] = np.mean(edit_d), np.mean(changed)
    return tf_lines, stat_d

def sanity_transform(args, hypo, mode, src_line = None, idx = None, transform_d = None):
    stat_d = {}
    
    #global NLP_TRF
    #if NLP_TRF is None:
    #    NLP_TRF = load_spacy_trf()
    spacytrf_d = {} # (jack 11/16) transform_d['spacy_trf_d'] #to avoid cuda conflict with comet, now i use processed spacytrf_result
    global INITIALIZED_PERTERB
    if not INITIALIZED_PERTERB: #could be a bug in checklist: if run for the first time, it could do nothing. 
        _ = Perturb.perturb([nlp('stub sentence')], Perturb.add_negation)
        INITIALIZED_PERTERB = True

    if mode.startswith('flu-randomworddrop-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split()
        drop_num = math.ceil(len(ww) * prob)
        if drop_num >= len(ww):
            drop_num = max(len(ww) - 1, 0)
        for kk in range(drop_num):
            drop_idx = random.randint(0, len(ww) - 1)
            ww = ww[:drop_idx] + ww[drop_idx + 1:]
        hypo_new = ' '.join(ww)
    
    if mode.startswith('lastwords-'):
        #lastwords-4-rep-2
        rep_num = int(mode.split('-')[-1])
        word_num = int(mode.split('-')[1])
        #if hypo[-1] == '.': #qualitatively, i found that deleting period gives less log-prob for bart_cnn
        #    hypo = hypo[:-1] #delete the last period
        rep_w = ' '.join(hypo.split(' ')[-word_num:])
        if rep_w.endswith('.'):
            rep_w = rep_w[0].upper() + rep_w[1:]
        hypo_new = hypo + (' ' + rep_w) * rep_num

    # if mode.startswith('con-shufflener'):
    #     '''
    #     random shuffle all NERs in a text paragraph
    #     '''
    #     global NLP
    #     if NLP is None:
    #         NLP = load_spacy()
    #     doc = NLP(text)
    #     ents_list = list(doc.ents)
    #     random.shuffle(ents_list)

    #     new_text = text
    #     for i, e in enumerate(reversed(doc.ents)): # reversed to not modify the offsets of other entities when substituting
    #         new_ner_str = ents_list[i].text
    #         new_text = new_text[:e.start_char] + new_ner_str + new_text[e.end_char:]

    #     return new_text

    if mode.startswith('con-replacesent'):
        sents = sent_tokenize(hypo)
        s_idx = random.choice(range(len(sents))) #don't use the idx variable!
        new_sent = random.choice(transform_d['all_sents'])
        while new_sent in hypo: new_sent = random.choice(transform_d['all_sents'])
        sents[s_idx] = new_sent
        hypo_new = ' '.join(sents)

    if mode.startswith('con-genericner'):
        hypo_new = transform_utils.generic_ner(hypo, transform_d)

    if mode.startswith('con-switchsent'):
        sents = sent_tokenize(hypo)
        if len(sents) >= 2:
            sents = [sents[1]] + [sents[0]] + sents[2:]
        hypo_new = ' '.join(sents)

    if mode.startswith('con-sentshuffle'):
        sents = sent_tokenize(hypo)
        random.shuffle(sents)
        hypo_new = ' '.join(sents)

    if mode.startswith('con-negate'):
        # IMPORTANT: about StopIteration runtime error: https://github.com/RaRe-Technologies/gensim/issues/2438#issuecomment-644753776
        # need to comment out ...(conda dir).../site-packages/pattern/text/__init__.py line 609
        # to get the conda lib path you can: from distutils.sysconfig import get_python_lib; print(get_python_lib())
        # /home/gridsan/tianxing/.conda/envs/metric/lib/python3.8/site-packages
        sents_new = []
        for sent in sent_tokenize(hypo):
            ret = Perturb.perturb([spacytrf_d[sent]], Perturb.add_negation)
             #' '.join([neg for pos,neg in ret['data']])
            if len(ret['data']) == 0:
                sent_negate = transform_utils.simple_negate(sent) #failed! fall back to naive negation heursitic...
                if sent_negate == sent: #still failed! just use the original one
                    logger.info('idx: %d, warning: sent_negate failed, using original sentence: %s', idx, sent)
            else:
                sent_negate = ret['data'][0][1]
            sent_negate = sent_negate.replace(' .', '.').replace('  ', ' ').strip() #sometimes it would add a space before the period
            assert(abs(len(sent.split()) - len(sent_negate.split())) <= 4) #make sure that 
            sents_new.append(sent_negate)
        hypo_new = ' '.join(sents_new)

    if mode.startswith('rep-repwhole-'):
        rep_num = int(mode.split('-')[-1])
        hypo_new = ' '.join([hypo] * (rep_num + 1))

    if mode.startswith('rep-lastsenrep-'):
        rep_num = int(mode.split('-')[-1])
        sens = sent_tokenize(hypo)
        last_s = sens[-1] 
        hypo_new = ' '.join([hypo] + [last_s] * rep_num)

    if mode.startswith('flu-removestopwords-'):
        prob = float(mode.split('-')[-1])
        doc = nlp(hypo); 
        delete_idx = []
        for token in doc:
            if token.is_stop == True:
                if random.random() <= prob:
                    #logger.info(token.text + ' ' + hypo)
                    if token.idx > 0 and hypo[token.idx - 1] == ' ':
                        delete_idx.append(token.idx - 1)    
                    for kk in range(token.idx, token.idx + len(token.text)):
                        delete_idx.append(kk)

        hypo_new = transform_utils.delete_str_idxs(hypo, delete_idx)
        hypo_new = hypo_new.replace('  ', ' ').strip()
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)        

    if mode.startswith('flu-truncate-'):
        ww = hypo.split()
        trunc_num = math.floor(len(ww) * float(mode.split('-')[-1]))
        ww = ww[:len(ww) - trunc_num]
        assert(len(ww) >= 1)
        hypo_new = ' '.join(ww)
        if hypo_new[-1] != '.': hypo_new += '.'

    if mode.startswith('flu-lemmatizeverb-'):
        # doc = spacytrf_d[hypo]; hypo_new = hypo;
        doc = nlp(hypo); hypo_new = hypo; # jack 11/16
        prob = float(mode.split('-')[-1])
        idx_shift = 0
        for token in doc:
            if token.pos_ in ['VERB']: #'VERB':
                if random.random() <= prob:
                    hypo_new = hypo_new[:token.idx + idx_shift] + token.lemma_ +  hypo_new[token.idx + idx_shift + len(token.text):]
                    idx_shift += len(token.lemma_) - len(token.text)
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)

    if mode.startswith('flu-removearticle-'):
        prob = float(mode.split('-')[-1])
        ww_new = []
        for w in hypo.split():
            to_add = True
            if w.lower() in ['a', 'an', 'the'] and random.random() <= prob: 
                to_add = False
            if to_add:
                ww_new.append(w)
        hypo_new = ' '.join(ww_new)
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)
    
    if mode.startswith('flu-removepreposition-'):
        #prep_lis = ['for', 'in', 'on', 'with', 'by', 'inside', 'outside']
        doc = nlp(hypo); hypo_new = hypo;
        remove_prob = float(mode.split('-')[-1])
        delete_idx = []
        for token in doc:
            if token.pos_ in ['ADP']: #'VERB':
                #logger.info(token.text + ' ' + token.pos_ + ' ' + hypo)
                if random.random() < remove_prob:
                    for kk in range(token.idx, token.idx + len(token.text)):
                        delete_idx.append(kk)
                    #hypo_new = hypo_new.replace(token.text, '', 1) #only replace one occurance
        hypo_new = transform_utils.delete_str_idxs(hypo, delete_idx)
        hypo_new = hypo_new.replace('  ', ' ').strip()
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)    

    if mode.startswith('flu-removepunct'):
        hypo_new = hypo.replace(',', '')
        hypo_new = hypo_new.replace('"', '')
        hypo_new = hypo_new.replace('.', '')
        hypo_new = hypo_new.replace(':', '')

    if mode.startswith('flu-noisepunct-'):
        prob = float(mode.split('-')[-1])
        hypo_new = ''
        t_d = {'.': ',', ',': '.', '!': ',', '?': ',', ':':'.'}
        for i in range(len(hypo)):
            if hypo[i] in ['.', ',', '!', '?', ':'] and random.random() <= prob:
                hypo_new += t_d[hypo[i]]
            else:
                hypo_new += hypo[i]

    if mode.startswith('flu-sentencemiddleswap-'):
        s_num = int(mode.split('-')[-1])
        hypo_new, stat_d = transform_utils.flu_sentencemiddleswap(hypo, s_num, stat_d)

    if mode.startswith('flu-randomlocalswap-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split(); assert(len(ww) >= 2)
        s_num = min(math.ceil(len(ww) * prob), len(ww) - 1)
        swaped = ['_PL_'] #want to avoid swap the same index twice
        for kk in range(s_num):
            ss_now = '_PL_'
            while ss_now in swaped:
                s_idx = random.randint(0, len(ww) - 2) #we will swap the token and the token after it
                ss_now = str(s_idx) + '_' + '_'.join([ww[s_idx], ww[s_idx + 1]])
            swaped.append(ss_now)
            ww = ww[:s_idx] + [ww[s_idx + 1]] + [ww[s_idx]] + ww[s_idx + 2:]
        hypo_new = ' '.join(ww)

    if mode.startswith('flu-randomtokenrep-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split()
        s_num = math.ceil(len(ww) * prob)
        for kk in range(s_num):
            s_idx = random.randint(0, len(ww) - 1) 
            ww = ww[:s_idx] + [ww[s_idx]] + [ww[s_idx]] + ww[s_idx + 1:]
        hypo_new = ' '.join(ww)

    if mode.startswith('flu-shufflewordinsent'):
        sens = sent_tokenize(hypo)
        s_id = random.randint(0, len(sens) - 1)
        sens[s_id] = transform_utils.shuffle_word_in_sent(sens[s_id])
        hypo_new = ' '.join(sens)

    if mode.startswith('alllower'):
        hypo_new = hypo.lower()

    if mode.startswith('addlastperiod'):
        hypo_new = hypo + '.'

    if mode.startswith('removelastperiod'):
        if not any([hypo.endswith(ww) for ww in ['.', '?', '!', '"', ':']]):
            logger.info('meet hypo not ending with period: %s', hypo)
            hypo_new = hypo
        else:
            hypo_new = hypo[:-1]

    if mode.startswith('refner-'):
        ref = transform_d['refs'][idx]
        ty = mode.split('-')[1]
        assert(ty == 'person')
        ref = ref.replace('Sir ', '').replace('Mr ', '').replace('Mr. ', '')
        while 1 == 1: #every time we modify the string, the index is changed, so we do "nlp" again
            doc = nlp(ref)
            found_new = False
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    replace_w = 'he'
                    pref = ref[max(ent.start_char - 11, 0):ent.start_char]
                    if any([(ww in pref) for ww in [' to ', ' with ', ' from ', ' over ', ' by ', ' beat ', ' defeat ', ' defeated ', ' shot ', ' down ']]):
                        replace_w = 'him'
                    if any([(ww in pref) for ww in [' the ']]):
                        replace_w = 'man'
                    if any([pref.endswith(ww) for ww in [' boss ', ' reporter ', ' no 1 ', ' striker ']]):
                        replace_w = ''
                    #if any([pref.lower().endswith(ww) for ww in ['sir ', 'mr ', 'mr. ']]):
                    #    replace_w = ent.text
                    if ent.text.endswith("'s"):
                        replace_w = 'his'
                    if pref.endswith('. ') or (ent.start_char == 0):
                        replace_w = replace_w[0].upper() + replace_w[1:]
                    ref_new = ref[:ent.start_char] + replace_w + ref[ent.end_char:]
                    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    #breakpoint()
                    ref = ref_new
                    found_new = True
                    break
            if found_new is False:
                break

        hypo_new = ref

    if mode.startswith('refspanrep-'):
        rep_num = int(mode.split('-')[1])
        #sen = sent_tokenize(src_line)[0]
        sen = transform_d['refs'][idx]
        #if sen[-1] == '.': sen = sen[:-1]
        rep_w = ' '.join(sen.split(' ')[-3:])
        if rep_w.endswith('.'):  #looks like both retaining the period and uppcase is useful!
            rep_w = rep_w[0].upper() + rep_w[1:]
        sen_rep = sen + (' ' + rep_w) * rep_num
        hypo_new = sen_rep

    if mode.startswith('srcsenrep-'):
        rep_num = int(mode.split('-')[1])
        sens = sent_tokenize(src_line)
        hypo_new = ' '.join([sens[0]] * rep_num)

    if mode.startswith('highfreqsource-'):
        top_num = int(mode.split('-')[1][3:])
        len_num = int(mode.split('-')[2][3:])
        src_tt = word_tokenize(src_line)
        freq_d = {}
        N_GRAM = 2
        for i in range(len(src_tt) - N_GRAM + 1):
            w = ' '.join(src_tt[i: i + N_GRAM])
            if not w in freq_d: freq_d[w] = 0
            freq_d[w] = freq_d[w] + 1
        w_d_sorted = sorted(freq_d.items(), key = lambda x: x[1], reverse = True)
        freq_w = [w[0] for w in w_d_sorted[:top_num]]
        gen_lis = []
        for i in range(len_num):
            gen_lis.append(freq_w[random.randint(0, top_num - 1)])
        hypo_new = ' '.join(gen_lis)

    if mode.startswith('highfreqrandom-'):
        top_num = int(mode.split('-')[1][3:])
        len_num = int(mode.split('-')[2][3:])
        w_d = transform_d['wfreq_d']
        w_d['.'], w_d[','] = -1000, -1000 #make sure they do not appear
        w_d_sorted = sorted(w_d.items(), key = lambda x: x[1], reverse = True)
        freq_w = [w[0] for w in w_d_sorted[:top_num]]
        gen_lis = []
        for i in range(len_num):
            gen_lis.append(freq_w[random.randint(0, top_num - 1)])
        
        #refs = transform_d['refs']
        hypo_new = ' '.join(gen_lis)

    if mode.startswith('refreservesort-'):
        sort_m, ref_id = mode.split('-')[-2], int(mode.split('-')[-1])
        if sort_m == 'freq': w_d = transform_d['wfreq_d']
        if sort_m == 'logprob': w_d = transform_d['wlogprob_d']
        ref_reserve = transform_d['refs_reserve'][idx]
        ref_scores = []
        for ref in ref_reserve:
            ref_tt = word_tokenize(ref.lower())
            score = []
            for w in ref_tt:
                if not w in w_d:
                    logger.info('warning: word [%s] not in w_d, skipping...', w)
                    continue
                score.append(w_d[w])
            score = np.mean(score)
            ref_scores.append((ref, score))
        ref_scores = sorted(ref_scores, key = lambda x: x[1], reverse = True)
        hypo_new = ref_scores[ref_id][0]
    
    if mode.startswith('refreserve-'):
        ref_id = int(mode.split('-')[-1])
        ref_reserve = transform_d['refs_reserve']
        hypo_new = ref_reserve[idx][ref_id]

    if mode.startswith('longestrefreserve-'):
        ref_id = int(mode.split('-')[-1])
        refs = transform_d['refs_reserve'][idx]
        refs = [(len(ww.split()), ww) for ww in refs]
        sorted_ref = sorted(refs, key = lambda x: x[0], reverse = True)
        hypo_new = sorted_ref[ref_id][1]

    if mode.startswith('useref'):
        refs = transform_d['refs']
        hypo_new = refs[idx]
        
    if mode.startswith('modelgen'):
        model_gens = transform_d['model_gens']
        hypo_new = model_gens[idx]

    if mode.startswith('copysrc'):
        hypo_new = src_line
     
    if mode.startswith('deletelastsen-'):
        del_num = int(mode.split('-')[-1])
        pos = [_.start() for _ in re.finditer('\.', hypo)] 
        if len(pos) <= 1:
            return hypo
        if len(pos) <= del_num:
            del_num = len(pos) - 1
        hypo_new = hypo[:pos[- del_num - 1] + 1]
 
    if mode.startswith('switchsentence'):
        pos = [_.start() for _ in re.finditer('\.', hypo)] 
        if len(pos) <= 1:
            return hypo
        pos = [-1] + pos
        if pos[-1] != len(hypo) - 1: pos = pos + [len(hypo) - 1]
        sens = [hypo[pos[l - 1] + 1: pos[l] + 1] for l in range(1, len(pos))]
        assert(''.join(sens) == hypo)
        if sens[1].startswith(' ') and not sens[0].startswith(' '):
            sens[0] = ' ' + sens[0]
        
        sp = random.randint(1, len(sens) - 1)
        sens[sp - 1], sens[sp] = sens[sp], sens[sp - 1]
        
        hypo_s = ''.join(sens)
        if hypo_s.startswith(' '): hypo_s = hypo_s[1:]
        hypo_new = hypo_s
        
    if mode.startswith('delduptoken'):
        dd_hypo = []
        for tt in hypo.split(' '):
            #if (tt in [',', '.']) or (not (tt in dd_hypo and tt in ['a', 'the', 'an', 'and'])):
            if (tt in [',', '.']) or (not (tt in dd_hypo)):
                dd_hypo.append(tt)
            #else: print(tt, ' ')
        hypo_new = ' '.join(dd_hypo)

    if args.debug_transform and idx <= 30:
        logger.info('idx: %d, hypo    : %s', idx, hypo)
        logger.info('idx: %d, hypo_new: %s', idx, hypo_new)
        logger.info('idx: %d, ref     : %s', idx, transform_d['refs'][idx])

    return hypo_new, stat_d

def op_identifier(args):
    suffix_str = ''
    return f'sanity_{args.operation}{suffix_str}'

if __name__ == '__main__':
    import argparse
    import os
    from util import load_file_by_line, write_file_by_line
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True, help='use relative path from project home dir!!')
    parser.add_argument('-op', '--operation', type=str, required=True) # i.e. mode in tx's code
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-f', '--force', action='store_true')

    # from tx
    parser.add_argument('--debug_transform', action='store_true')

    args = parser.parse_args()
    assert not args.generation[0].startswith('/'), 'use relative path from project home dir!!'

    for gen in args.generation:
        new_path = os.path.join('gen_mod', op_identifier(args), gen[4:]) # ignore gen/ in front
        assert not os.path.exists(new_path) or args.force, f'output path {new_path} already exist, stopped if not forcing!!'
        texts = load_file_by_line(gen)

        transform_d = {'src': ['STUB' for _ in range(len(texts))]} # TODO: change this FAKE alternative reference
        warped_texts, stat_d = batch_sanity_transform(args, texts, args.operation, transform_d, seed=args.seed)

        stats_repr = [f'#\t{k}\t{v}' for k,v in stat_d.items()]
        outputs = stats_repr + warped_texts
        write_file_by_line(new_path, outputs)