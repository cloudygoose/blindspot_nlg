import os, sys, math
import nltk
import numpy as np
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import copy
import logging
#from myutil import MyTimer
import torch
import random
from tqdm import tqdm
import time, collections
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.getLogger()

try: 
    from multiprocessing import cpu_count
except: 
    from os import cpu_count

def eval_gen(args, refs, gens, gens_full, do_save = False, save_ss = ''):
    #bleu_score_2 = text_eval.ref_bleu(refs, gens, 2)
    #bleu_score_3 = text_eval.ref_bleu(refs, gens, 3)
    g_ppl = gpt_ppl(args, gens_full)
    logger.info('gpt_ppl: %f', g_ppl)

    #entropy_score = text_entropy(gens, 3)
    #repeat_r = repeat_ratio(gens, num_gram = 4, thres_time = 3)
    rep_4 = rep_ngram(gens, num_gram = 4)
    rep_3 = rep_ngram(gens, num_gram = 3)
    rep_2 = rep_ngram(gens, num_gram = 2)
    rep_1 = rep_ngram(gens, num_gram = 1)
    #logger.info('rouge: %s entropy: %f', str(rouge_res), entropy_score)
    logger.info('rep_4: %.3f rep_3: %.3f rep_2: %.3f rep_1: %.3f', rep_4, rep_3, rep_2, rep_1)

    logger.info('computing bleu...')
    bleu_score = corpus_ref_bleu(refs, gens, 3)
    #rouge_res = ref_rouge(refs, gens)
    self_bleu_score = corpus_self_bleu(gens, 3)
    logger.info('bleu_3: %.3f self_bleu_3: %.3f', bleu_score, self_bleu_score)
 
    res = {'bleu_score': bleu_score, 'self_bleu_score': self_bleu_score, 'rep_4': rep_4, 'rep_3': rep_3, 'rep_2': rep_2, 'rep_1' : rep_1, 'gpt_ppl': g_ppl}   
    if do_save:
        save_fn = args.work_dir + f'/eval_gen/eval_gen_{save_ss}.save'
        res.update({'refs': refs, 'gens': gens, 'gens_full': gens_full})
        logger.info('saving eval res to %s', save_fn)
        torch.save(res, save_fn)
    return res

def gpt_ppl(args, gens):
    PPL_TOKENIZER_STRING = 'EleutherAI/gpt-neo-2.7B'
    PPL_MODEL_STRING = 'EleutherAI/gpt-neo-2.7B'
    
    logger.info('loading gpt-neo...')

    tokenizer = AutoTokenizer.from_pretrained(PPL_TOKENIZER_STRING, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_STRING, local_files_only = True)
    model = model.cuda()

    nlls, p_nlls, lengths = [], [], [] #nll with penalty
    loss_ce = CrossEntropyLoss(reduction = 'none')

    ind_ = torch.LongTensor(list(range(1))).cuda() #bz = 1
    for text in tqdm(gens, desc='perplexity'):
        input_ids = tokenizer.encode(text, return_tensors='pt').cuda()
        target_ids = input_ids.clone()
        outputs = model(input_ids, labels=target_ids); lm_logits = outputs[1]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.reshape(shift_labels.size())

        """
        shift_logits = torch.log_softmax(shift_logits, dim = -1) #this is to ensure all logits are negative
        p_mask = torch.zeros(shift_logits.size()).cuda() #penalty mask
        for j in range(1, shift_labels.size(1)):
            p_mask[:, j] = p_mask[:, j - 1] #his is incremental
            p_mask[ind_, j, shift_labels[:, j - 1]] += 1.0
        p_loss = loss_ce((shift_logits - p_mask).view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        p_loss = p_loss.reshape(shift_labels.size())
        p_nlls.append(p_loss[0, args.prefix_len - 1:].sum().item())
        """

        nlls.append(loss[0, args.prefix_len - 1:].sum().item())

        lengths.append(loss[0, args.prefix_len - 1:].size(0))
    
    res = {}
    res['avg_ppl'] = math.exp(np.sum(nlls) / np.sum(lengths))
    #res['avg_ppl_penalty'] = math.exp(np.sum(p_nlls) / np.sum(lengths))

    return res['avg_ppl']

def rep_ngram(sen_lis, num_gram = 4):
    rep_lis = []
    for sen in sen_lis:
        uniq_ngram, all_ngram = {}, []
        for i in range(0, len(sen) - num_gram + 1):
            tt = ' '.join(sen[i:i + num_gram])
            if not tt in uniq_ngram: uniq_ngram[tt] = True
            all_ngram.append(tt)
        if len(all_ngram) == 0:
            logger.info('warning: len(all_ngram) is 0!!! skipping... sample: %s', str(sen))
            continue
        rep = 1.0 - len(uniq_ngram) * 1.0 / len(all_ngram)
        rep_lis.append(rep)
    return np.mean(rep_lis)

def repeat_ratio(sen_lis, num_gram = 3, thres_time = 3):
    repeat_lis = []
    for sen in sen_lis:
        r = 0
        dd = collections.defaultdict(lambda: 0)
        for i in range(0, len(sen) - num_gram + 1):
            tt = ' '.join(sen[i:i + num_gram])
            dd[tt] += 1
            if dd[tt] >= thres_time:
                r = 1
        repeat_lis.append(r)
    return np.mean(repeat_lis)

def text_entropy(sen_lis, k):
    #sen_lis is like [['i','am','you','</s>'] ...]
    #assume it is lowered case, and clean
    dd, num = {}, 0
    for sen in sen_lis:
        for i in range(0, len(sen) - k + 1):
            num += 1
            tt = ' '.join(sen[i:i+k])
            #print tt
            if not tt in dd: dd[tt] = 0
            dd[tt] += 1
    
    entro = 0.0
    for tt in dd:
        prob = float(dd[tt] * 1.0) / num
        entro = entro - math.log(prob) * prob
    return entro
    
def ex_text_entropy():
    s1 = 'i am you </s>'.split()
    s2 = 'hello guys </s>'.split()
    s3 = 'come on </s>'.split()
    s4 = 'i like you </s>'.split()
    ss = [s1, s2, s3, s4]
    print(text_entropy(ss, 2))

def corpus_ref_bleu(refs, samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    """
    scores = []
    for kk, s in enumerate(samples):
        print(kk)
        if kk % int(len(samples) / 100) == 0 and kk > 0: 
            print(kk * 1, 'percent')
        score = nltk.translate.bleu_score.sentence_bleu(refs, s, weights, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return np.mean(scores)
    """
    refs_lis = [refs] * len(samples)
    score = nltk.translate.bleu_score.corpus_bleu(refs_lis, samples, weights, smoothing_function=SmoothingFunction().method1)
    return score

def ref_bleu(refs, samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    score = nltk.translate.bleu_score.corpus_bleu(refs, samples, weights, smoothing_function=SmoothingFunction().method1)
    return score

def ref_rouge(refs, samples):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    assert(len(refs) == len(samples))
    r2_lis, rl_lis = [], []
    max_rl, max_rl_s = 0, None
    for ref, sample in zip(refs, samples):
        if isinstance(ref, list) and isinstance(ref[0], list) and len(ref) == 1:
            ref = ref[0]
        ss = scorer.score(' '.join(ref), ' '.join(sample))
        r2_lis.append(ss['rouge2'].fmeasure)
        rl = ss['rougeL'].fmeasure
        rl_lis.append(rl)
        if rl > max_rl:
            max_rl, max_rl_s = rl, (ref, sample)
    res = {'rouge2': np.mean(r2_lis), 'rougeL': np.mean(rl_lis)}
    return res

def corpus_self_bleu(samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    """
    scores = []
    for kk, s in enumerate(samples):
        print(kk)
        if kk % int(len(samples) / 100) == 0 and kk > 0: 
            print(kk * 1, 'percent')
        score = nltk.translate.bleu_score.sentence_bleu(refs, s, weights, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return np.mean(scores)
    """
    refs_lis = []
    for i in range(len(samples)):
        refs_lis.append(samples[:i] + samples[i+1:])
    score = nltk.translate.bleu_score.corpus_bleu(refs_lis, samples, weights, smoothing_function=SmoothingFunction().method1)
    return score

def calc_bleu(refs, s, weights):
    return nltk.translate.bleu_score.sentence_bleu(refs, s, weights, smoothing_function=SmoothingFunction().method1)
 
def calc_meteor(refs, s):
    return meteor_score(refs, s)

#import nist_code
def calc_nist(refs, s, ngram):
    return nist_code.sentence_nist(refs, s, ngram)

def ex_nist_play():
    hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'
    
    reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    reference3 = 'It is the practical guide for the army always to heed the directions of the party'
    print('nist_score:', calc_nist([reference1, reference2, reference3], hypothesis1, 3))

#ex_nist_play()

def corpus_bleu_parallel(refs, samples, num_gram, do_meteor = False, do_nist = False):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    
    cpu_num = min(cpu_count(), 35)
    print('cpu_num used:', cpu_num)
    
    scores = []
    pool = Pool(cpu_num) #Pool(cpu_count() - 5)
    result = list()
    pp = 1
    if do_meteor or do_nist:
        refs = [' '.join(ll) for ll in refs]
        samples = [' '.join(ll) for ll in samples]

    for kk, s in enumerate(samples):
        if do_meteor == True:
            result.append(pool.apply_async(calc_meteor, args=(refs, s)))
        elif do_nist == True:
            result.append(pool.apply_async(calc_nist, args=(refs, s, num_gram)))
        else:
            result.append(pool.apply_async(calc_bleu, args=(refs, s, weights)))
        if kk % int(len(samples) / 5) == 0 and kk > 0: 
            for it, i in enumerate(result):
                scores.append(i.get())
            pool.close()
            pool.join()
            pool = Pool(cpu_num) #Pool(cpu_count() - 5)
            result = list()
            print(pp * 20, 'percent')
            pp += 1
    
    if len(result) > 0: 
        for it, i in enumerate(result):
            scores.append(i.get())
        pool.close()
        pool.join()
    
    print('finished, len(scores):', len(scores))
    return np.mean(scores), np.std(scores)

def corpus_self_bleu_parallel(samples, num_gram):
    if num_gram == 2:
        weights = (0.5, 0.5)
    if num_gram == 3:
        weights = (0.333333, 0.333333, 1 - 0.333333 * 2)
    if num_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    
    print('cpu_count', cpu_count())
    pool = Pool(cpu_count() - 5)
    result = list()
 
    for i, s in enumerate(samples):
        result.append(pool.apply_async(calc_bleu, args=(samples[:i] + samples[i+1:], s, weights)))
    
    scores = []
    for it, i in enumerate(result):
        scores.append(i.get())

    pool.close()
    pool.join()
    return np.mean(scores), np.std(scores)


def ex_corpus_bleu():
    samples = [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'f']]
    refs = [['a', 'b', 'd'], ['b', 'c', 'f'], ['d', 'f', 'e']]
    print(corpus_bleu(refs, samples, 2))

def ex_meteor_play():
    hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'
    
    reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    reference3 = 'It is the practical guide for the army always to heed the directions of the party'

    print('eeteor_score:', meteor_score([reference1, reference2, reference3], hypothesis2))

def eval_lm_corpus_bleu(model, ref_fn, filter_len, sample_num, do_log = True, jump_ref = 0):
    if do_log == True: logger.info('loading refs %s', ref_fn)
    refs = []
    l_count = 0
    for l in open(ref_fn, 'r').readlines():
        l_count = l_count + 1
        if l_count < jump_ref: continue
        ss = l.lower().split()
        if len(ss) < filter_len: continue
        if ss[0] == '<s>': ss = ss[1:]
        if ss[-1] == '</s>': ss = ss[:-1]
        if len(ss) < filter_len: continue
        
        refs.append(ss[:filter_len])        
        if len(refs) == sample_num: break
    
    assert(len(refs) == sample_num)
    
    timer = MyTimer(name = 'model sampling')
    timer.tic()    
    samples = []
    for l in range(sample_num // 50):
        b_samples, _ = model.sampleBatch(filter_len, 50, full_length = True, give_id = False)
        samples.extend(b_samples)
    assert(len(samples) == sample_num)
    timer.toc()
    timer.report()

    timer = MyTimer(name = 'bleu')
    timer.tic()    
    res = {
        'bleu-2': corpus_bleu_parallel(refs, samples, 2),
        'self-bleu-2': corpus_bleu_parallel(samples, refs, 2),
        'bleu-3': corpus_bleu_parallel(refs, samples, 3),
        'self-bleu-3': corpus_bleu_parallel(samples, refs, 3)
    }
    timer.toc()
    timer.report()

    return res

def tvd(a, b):
    return torch.sum(torch.abs(a - b)).item() / 2

def filter_len(lis, request_len, filter_len, start_idx):
    res = []
    while (len(res) < request_len or request_len == -1) and start_idx < len(lis):
        while len(lis[start_idx]) <= filter_len: 
            start_idx = start_idx + 1
        res.append(lis[start_idx])
        start_idx = start_idx + 1
    return res, start_idx

def read_file(fn, vocab_inv, filter_len = -1, ret = 'idx'):
    logger.info('reading sample_prefix_fn %s', fn)
    p_idx = open(fn, 'r').readlines()
    p_idx = [s.strip().lower().split() for s in p_idx]
    res = []
    for i in range(len(p_idx)):
        l = p_idx[i]
        assert(l[0] != '<s>' and l[-1] != '</s>')
        l = [(vocab_inv[w] if w in vocab_inv else vocab_inv['<unk>']) for w in l]
        res.append(l)
    return res

def tvd_batch(a, b):
    return torch.sum(torch.abs(a - b), dim = 1) / 2

def kl_batch(a, b):
    return torch.sum(a * (torch.log(a) - torch.log(b)), dim = 1)
    #return torch.sum(torch.abs(a - b), dim = 1) / 2

def js_batch(a, b):
    m = (a + b) * 0.5
    return 0.5 * (kl_batch(a, m) + kl_batch(b, m))

def gd_batch(a, b):
    v1, idx1 = torch.max(a, 1)
    v2, idx2 = torch.max(b, 1)
    dd = (1 - torch.FloatTensor((idx1 == idx2).numpy()))
    return dd
 
def get_kth_cond_dis_pytorch(num, kth, sample_m, m, measure = 'tvd', corrupt_ratio = 0):
    bz = 50
    dis_lis = []
    for k in range(num // bz + 1):
        samples, logp = sample_m.sampleBatch(kth + 1, bz, full_length = True) 
        if corrupt_ratio > 0:
            for i in range(bz):
                for j in range(len(samples[i])):
                    if random.random() < corrupt_ratio:
                        samples[i][j] = random.randint(4, m.output_size - 1)
        output_m = m.sampleBatch_prefix(kth + 1, bz, kth, samples, only_kth_outputdistro = kth)
        output_sm = sample_m.sampleBatch_prefix(kth + 1, bz, kth, samples, only_kth_outputdistro = kth)
        #print(output_m.size(), output_sm.size())
        if measure == 'tvd':
            dis = tvd_batch(output_m, output_sm)
        elif measure == 'js':
            dis = js_batch(output_m, output_sm)
        elif measure == 'gd':
            dis = gd_batch(output_m, output_sm)
        dis_lis.append(dis.mean().item())
    return np.mean(dis_lis)

def eval_cond_exposurebias(num, data_m, m, seq_len, measure = 'tvd', start_seq_len = 1):
    eval_stat = {}
    print('eval_sample_num:', num, 'measure:', measure)
    for k in range(start_seq_len, seq_len):
        data_sample_tvd = get_kth_cond_dis_pytorch(num, k, data_m, m, measure)
        model_sample_tvd = get_kth_cond_dis_pytorch(num, k, m, data_m, measure)
        eval_stat[k] = {
            'CGP(P_M|P_D)': data_sample_tvd,
            'CGP(P_M|P_M)': model_sample_tvd,
            'ratio': model_sample_tvd / data_sample_tvd,
        }
        #for c in [0.3, 0.6, 0.9]: 
        #    eval_stat[k]['CGP(P_M|P_M_corrupt' + str(c) + ')'] = get_kth_cond_dis_pytorch(num, k, m, data_m, measure, c)
        print(k, eval_stat[k], time.asctime( time.localtime(time.time()) ))
    return eval_stat

def eval_lm_exposure_bias(rnn_glo, ref_data_fn, ref_prefix_fn, vocab_inv, EVAL_SAMPLE_RANDOM_SEED, EVAL_SAMPLE_NUM, SEQ_LEN, BZ = 50):
    logger.info('EVAL_SAMPLE_RANDOM_SEED: %d', EVAL_SAMPLE_RANDOM_SEED)
    torch.manual_seed(EVAL_SAMPLE_RANDOM_SEED)
    torch.cuda.manual_seed(EVAL_SAMPLE_RANDOM_SEED)
    np.random.seed(seed = EVAL_SAMPLE_RANDOM_SEED)
    random.seed(EVAL_SAMPLE_RANDOM_SEED) 

    vocab = rnn_glo.vocab
    
    res = {}
    eb_rate_lis = []
    for EVAL_POS in range(1, SEQ_LEN):
        logger.info('Doing EVAL_POS: %d', EVAL_POS)
        logger.info('loading refs %s', ref_data_fn)
        
        #get data distribution at position EVAL_POS
        data_distro = torch.zeros(len(vocab)) #np.array([0 for kk in range(len(vocab))])
        refs = open(ref_data_fn, 'r').readlines()
        refs = [s.strip().lower().split() for s in refs]
        refs, _ = filter_len(refs, -1, SEQ_LEN, 0)
        for l in refs:
            assert(l[0] != '<s>' and l[-1] != '</s>')
            w = l[EVAL_POS]
            if not w in vocab_inv: w = '<unk>'
            data_distro[vocab_inv[w]] += 1
        refs = refs[:EVAL_SAMPLE_NUM]
        logger.info('data distribution approximated from %d ref sentences.', len(refs))
        data_distro = data_distro / torch.sum(data_distro).item()
        
        logger.info('getting model_distro EVAL_SAMPLE_NUM: %d', EVAL_SAMPLE_NUM)
        BN = BZ
        p_idx = read_file(ref_prefix_fn, vocab_inv)
        p_idx, _ = filter_len(p_idx, -1, SEQ_LEN, 0)
        random.shuffle(p_idx)
        model_distro = torch.zeros(len(vocab))
        model_distro_pre = torch.zeros(len(vocab))
        
        for k in range(int(EVAL_SAMPLE_NUM / BN)):
            b_dis = rnn_glo.sampleBatch(SEQ_LEN, BN, full_length = True, only_kth_outputdistro = EVAL_POS)
            pre = p_idx[(k * BN):((k + 1) * BN)]
            assert(len(pre) == BN)
            b_dis_pre = rnn_glo.sampleBatch_prefix(SEQ_LEN, BN, EVAL_POS, pre, full_length = True, only_kth_outputdistro = EVAL_POS)
            
            for l in range(BN):
                model_distro = model_distro + b_dis[l]
                model_distro_pre = model_distro_pre + b_dis_pre[l]

        model_distro = model_distro / EVAL_SAMPLE_NUM
        model_distro_pre = model_distro_pre / EVAL_SAMPLE_NUM
        logger.info('tvd model_distro %f', tvd(model_distro, data_distro))
        logger.info('tvd model_distro_pre %f', tvd(model_distro_pre, data_distro))
        
        logger.info('tvd model_distro - tvd model_distro_pre %f', tvd(model_distro, data_distro) - tvd(model_distro_pre, data_distro))
        logger.info('tvd model_distro / model_distro_pre %f', tvd(model_distro, data_distro) / tvd(model_distro_pre, data_distro))
        eb_rate_lis.append(tvd(model_distro, data_distro) / tvd(model_distro_pre, data_distro)) 
        res[EVAL_POS] = {
            'model_distro': model_distro.cpu(),
            'data_distro': data_distro.cpu(),
            'model_distro_pre': model_distro_pre.cpu(),
        }
        #for k in range(20):
        #    print(k, vocab[k], ':', data_distro[k])
    logger.info('mean eb_rate: %f', np.mean(eb_rate_lis), np.std())
    return res 

#ex_text_entropy()
#ex_corpus_bleu()

