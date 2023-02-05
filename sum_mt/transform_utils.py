import re
import logging, random, math
import numpy as np
from nltk import word_tokenize, sent_tokenize
import spacy
import pyinflect
import utils

nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger()

def shuffle_word_in_sent(s):
    tts = s.split()
    if len(tts) <= 4:
        return s
    t_h, t_l = tts[0], tts[-1]
    t_body = tts[1:-1]
    random.shuffle(t_body)
    s_shuf = ' '.join([t_h] + t_body + [t_l])
    return s_shuf

#shuffle_word_in_sent('Hi, i am you, while you have a big head, right?')
def con_switch_ner(text, num, spacytrf_d):
    '''
    random shuffle all NERs in a text paragraph
    '''
    
    assert isinstance(num, int)
    doc = spacytrf_d[text]
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

def con_switch_noun(text, num, spacytrf_d):
    assert isinstance(num, int)

    doc = spacytrf_d[text]

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


def con_switch_verb(text, num, spacytrf_d, NLP):

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
                    #print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... skipping')
                    #continue
                    print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... using original form instead')
                    i_token_j_infl = i_token # failure protection
                j_token = tokens[j_pos].strip()
                j_token_i_infl = NLP(j_token)[0]._.inflect(i_type)
                if j_token_i_infl is None:
                    #print(f'warning: failed to inflect [{i_token}]({i_type}) to {j_type}... skipping')
                    #continue
                    print(f'warning: failed to inflect [{j_token}]({j_type}) to {i_type}... using original form instead')
                    j_token_i_infl = j_token # failure protection
                tokens[i_pos] = tokens[i_pos].replace(i_token, j_token_i_infl)
                tokens[j_pos] = tokens[j_pos].replace(j_token, i_token_j_infl)
                ###
                last_i, last_j = i, j
                success += 1
    new_text = ''.join(tokens)
    return new_text


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
    # end of function lcs

def uppercase_sent_begin(hypo):
    sents = sent_tokenize(hypo); 
    for i in range(len(sents)):
        sents[i] = sents[i][0].upper() + sents[i][1:]
    return ' '.join(sents)

def delete_str_idxs(hypo, idxs):
    hypo_new = ''
    for kk in range(len(hypo)):
        if not kk in idxs:
            hypo_new += hypo[kk]
    return hypo_new

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

def con_generic_ner(text, s_prob, transform_d):
    '''
    replace NER of a particular type with generic names
    '''
    spacytrf_d = transform_d['spacy_trf_d']
    doc = spacytrf_d[text]

    new_text = text
    ner_types = set()
    for e in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        if not e.label_ in NER_TRANSFORM.keys(): continue
        if not (random.random() < s_prob): continue
        new_text = new_text[:e.start_char] + e.label_ + new_text[e.end_char:]
        ner_types.add(e.label_)

    #htx: i add this to delete articles before inserted label_
    tt = new_text.split(); new_tt = []
    for i, ww in enumerate(tt):
        if not (ww.lower() in ['a', 'an', 'the'] and i < len(tt) - 1 and tt[i + 1] in ner_types): new_tt.append(ww)
    new_text = ' '.join(new_tt)

    # __import__('pdb').set_trace()
    for ner_type in ner_types:
        if not ner_type in NER_TRANSFORM.keys(): continue
        start_idx = new_text.find(ner_type)
        while start_idx != -1:
            new_ent_entry = ''
            pref, pref_all = new_text[max(0, start_idx-MAX_ARTICLE_LENGTH-2):start_idx], new_text[:start_idx]
            allowed_articles, entry = NER_TRANSFORM[ner_type]
            nlp_pref_all = nlp(pref_all)
            if (not any(f' {art} ' in pref for art in ARTICLES)) and (not (len(nlp_pref_all) >= 1 and nlp_pref_all[-1].pos_ in ['ADJ', 'NOUN'])): # f' {art} ' to parse full word
                new_ent_entry += random.choice(allowed_articles) + ' '
            new_ent_entry += entry
            if any(f'{p} ' in pref[-3:] for p in EOS_PUNCT + ['\\n']):
                # start of sentence
                new_ent_entry = new_ent_entry.capitalize()
            new_text = new_text[:start_idx] + new_ent_entry + new_text[start_idx+len(ner_type):]

            start_idx = new_text.find(ner_type)

    new_text = uppercase_sent_begin(new_text)
    # __import__('pdb').set_trace()
    return new_text

def simple_negate(hypo):
    if ' not ' in hypo:
        return hypo.replace(' not ', ' ')
    if " didn't " in hypo:
        return hypo.replace(" didn't ", ' did ')  
    if " don't " in hypo:
        return hypo.replace(" don't ", ' do ')
    if " doesn't " in hypo:
        return hypo.replace(" doesn't ", ' does ')   
    if " cannot " in hypo:
        return hypo.replace(" cannot ", ' can ')
    if " can't " in hypo:
        return hypo.replace(" can't ", ' can ')
    if " won't " in hypo:
        return hypo.replace(" won't ", ' will ')
    if ' never ' in hypo:
        return hypo.replace(" never ", ' ')
    if ' were ' in hypo and (not ' were not ' in hypo):
        return hypo.replace(" were ", ' were not ')
    if ' are ' in hypo and (not ' are not ' in hypo):
        return hypo.replace(" are ", ' are not ')        
    return hypo

def flu_sentencemiddleswap(hypo, s_num, stat_d):
    sents = sent_tokenize(hypo); 
    #s_id, max_sl = -1, -1
    #for i, sent in enumerate(sents):
    #    if len(word_tokenize(sent)) > max_sl:
    #        max_sl = len(word_tokenize(sent))
    #        s_id = i  
    s_num = min(len(sents), s_num)
    swapped = [] #avoid swapping the same sentence
    
    for kk in range(s_num):
        s_id = random.randint(0, len(sents) - 1)
        while s_id in swapped:
            s_id = random.randint(0, len(sents) - 1)
        swapped.append(s_id)
        
        ww = word_tokenize(sents[s_id]); 
        if len(ww) <= 1: continue
        mid_pos = math.ceil(len(ww) * 1.0 / 2)
        if ww[0].lower() in ['the', 'in', 'a', 'on', 'at', 'around', 'when', 'and']:
            ww[0] = ww[0].lower()
        if ww[-1] == '.':
            ww_swap = ww[mid_pos:-1] + ww[0:mid_pos] + [ww[-1]]
        else:
            ww_swap = ww[mid_pos:] + ww[:mid_pos]
        if ww_swap[0] == ',': ww_swap = ww_swap[1:]
        if ww_swap[-2] == ',' and ww_swap[-1] == '.': ww_swap = ww_swap[:-2] + [ww_swap[-1]]
        sent_new = utils.nltk_detokenize(' '.join(ww_swap))
        sent_new = sent_new[0].upper() + sent_new[1:]
        sents[s_id] = sent_new

    return ' '.join(sents), stat_d


def compute_ngram_freq(lines):
    resd = {}
    for ngram in [3, 4]:
        dd = {}
        for line in lines:
            tt = line.lower().strip().replace('  ', ' ').split()
            for i in range(len(tt)):
                if i + ngram >= len(tt) + 1: continue
                ng = ' '.join(tt[i : i + ngram])
                if not ng in dd: dd[ng] = 0
                dd[ng] += 1
        ld =  sorted(dd.items(), key = lambda x: x[1], reverse = True)
        resd['{}gram'.format(ngram)] = ld
    return resd