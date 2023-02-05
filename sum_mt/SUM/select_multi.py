import argparse
import os, sys, torch, time
from pathlib import Path
import time, random, json, math
import numpy as np
from utils import *

#sys.path.append('/home/gridsan/tianxing/txml_shared/projects/metricnlg_2205/BARTScore')
sys.path.append(str(Path(__file__).absolute().parent.parent))
import sanity_transform

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(); logger.handlers = [];
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

SRC_HYPO = read_file_to_list('files/src_hypo_prompt.txt')
REF_HYPO = read_file_to_list('files/ref_hypo_prompt.txt')


class Scorer:
    """ Support ROUGE-1,2,L, BERTScore, MoverScore, PRISM, BARTScore """

    def __init__(self, args, file_path, device='cuda:0', multi_ref=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including src, ref_summ, ref_summs, and sys_summ.
        """
        self.multi_ref = multi_ref
        self.device = device
        self.data = read_pickle(file_path)
        print(f'Data loaded from {file_path}.')
        
        self.args = args
        if args.use_select_multi:
            self.sys_names = ['human', 'M0']
            load_fn = 'select_multiref/best_ref_comet.save'
            #load_fn = 'select_multiref/best_ref_rouge.save'
            #load_fn = 'select_multiref/best_ref_bert_score.save'
            #load_fn = 'select_multiref/best_ref_bart_score_cnn.save'
            #load_fn = 'select_multiref/best_ref_mover_score.save'
            logger.info('loading human gen from %s', load_fn); time.sleep(2)
            ld = torch.load(load_fn)
            for idx, doc_id in enumerate(self.data):
                self.data[doc_id]['sys_summs']['human'] = {'sys_summ': ld[idx], 'scores': {}}
        else:
            self.sys_names = self.get_sys_names()

        self.multi_ref_lines, self.multi_ref_lines_reserve = self.get_multi_ref_lines()

        if not multi_ref:
            self.single_ref_lines = self.get_single_ref_lines()
            print(f'In a single-reference setting.')
        else:
            self.ref_num = len(self.multi_ref_lines[0])
            print(f'In a multi-reference setting.')
            self.single_ref_lines = self.get_single_ref_lines()
        
        ld_fn = '../plm_ft/cnndm_saves/wfreq.save'
        logger.info('loading wfreq from %s', ld_fn)
        freq_d = torch.load(ld_fn)
        f_sum = sum([w[1] for w in freq_d.items()])
        logprob_d = {w: math.log(v * 1.0 / f_sum) for w,v in freq_d.items()}
        self.wfreq_d, self.wlogprob_d = freq_d, logprob_d

    def get_sys_names(self):
        first_id = list(self.data.keys())[0]
        return list(self.data[first_id]['sys_summs'].keys())

    def get_single_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summ'])

        return ref_lines

    def get_multi_ref_lines(self):
        ref_lines = []; ref_lines_reserve = [];
        #for doc_id in self.data:
        #    ref_lines.append(self.data[doc_id]['ref_summs']) #this has 11 ref, because it contains the original ref!
        
        mul_d = {}; ld_fn = './THumB/cnndm/cnndm_references.jsonl';
        logger.info('loading multi references from %s', ld_fn) #the original cnndm reference is not in this set
        with open(ld_fn, 'r') as json_file:
            for j_str in list(json_file):
                ss = json.loads(j_str); 
                assert(len(ss['refs']) == 10)
                mul_d[ss['seg_id']] = ss['refs'];
        
        R_NUM = 10
        logger.info('IMPORTANT: only using the last %d as reference!', R_NUM)
        for doc_id in self.data:
        #for i in range(len(ref_lines)):
            rr = mul_d[doc_id.split('-')[2]]
            #ref_lines.append(rr[10 - R_NUM:])
            ref_lines_reserve.append(rr[:10 - R_NUM])
            ref_lines.append(rr[(10 - R_NUM):])
            assert(len(ref_lines[-1]) == R_NUM and len(ref_lines_reserve[-1]) == 10 - R_NUM)

        return ref_lines, ref_lines_reserve

    def get_sys_lines(self, sys_name):
        sys_lines = []
        for doc_id in self.data:
            sys_lines.append(self.data[doc_id]['sys_summs'][sys_name]['sys_summ'])
        return sys_lines

    def get_src_lines(self):
        src_lines = []
        for doc_id in self.data:
            src_lines.append(self.data[doc_id]['src'])
        return src_lines

    def save_data(self, path):
        save_pickle(self.data, path)

    def score(self, metrics):
        args = self.args
        """ metrics: list of metrics """
        for metric_name in metrics:
            if metric_name == 'bert_score':
                from bert_score import BERTScorer

                # Set up BERTScore
                bert_scorer = BERTScorer(
                    lang='en',
                    idf=False,
                    rescale_with_baseline=True,
                    device=self.device
                )
                logger.info(f'BERTScore setup finished. Begin calculating BERTScore.')
                if args.hypo_transform is not None: 
                    logger.info('will apply transform %s', args.hypo_transform)
                start = time.time()
                ref_lines = self.single_ref_lines if not self.multi_ref else self.multi_ref_lines
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in ref_lines]
                else:
                    ref_lines = [[detokenize(line) for line in ww] for ww in ref_lines]

                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve}
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    if args.hypo_transform is not None:
                        sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, transform_d = transform_d, idx = kk) for kk, line in enumerate(sys_lines)] 
                    sys_lines = [detokenize(line) for line in sys_lines]

                    if not self.multi_ref:
                        P, R, F = bert_scorer.score(sys_lines, ref_lines)
                    else:
                        total_num = len(sys_lines)
                        P, R, F = np.zeros(total_num), np.zeros(total_num), np.zeros(total_num)
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_P, curr_R, curr_F = bert_scorer.score(sys_lines, ref_list)
                            P += curr_P.numpy()
                            R += curr_R.numpy()
                            F += curr_F.numpy()
                        P, R, F = P / self.ref_num, R / self.ref_num, F / self.ref_num
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            'bert_score_p': P[counter],
                            'bert_score_r': R[counter],
                            'bert_score_f': F[counter]
                        })
                        counter += 1
                logger.info(f'Finished calculating BERTScore, time passed {time.time() - start}s.')
                for variant in ['r', 'p', 'f']:
                    sys_scores = {}
                    mn = 'bert_score' + '_' + variant
                    for doc_id in self.data:
                        sys_summs = self.data[doc_id]['sys_summs']
                        for sys_name in self.sys_names:
                            if not sys_name in sys_scores: sys_scores[sys_name] = []
                            sys_scores[sys_name].append(sys_summs[sys_name]['scores'][mn])

                    avg_scores = []
                    for sys_name in sys_scores:
                        avg_s = np.mean(sys_scores[sys_name])
                        logger.info('%s len: %d avg: %f', sys_name, len(sys_scores[sys_name]), avg_s)
                        avg_scores.append(avg_s)
                    logger.info('===avg scores for %s: %f median: %f', mn, np.mean(avg_scores), np.median(avg_scores))

            elif metric_name == 'mover_score':
                from moverscore import word_mover_score, get_idf_dict

                # Set up MoverScore
                with open('files/stopwords.txt', 'r', encoding='utf-8') as f:
                    self.stop_words = set(f.read().strip().split(' '))

                #htx: i will calculate idf for each system
                # IDF for all system hypos, used for MoverScore
                #self.sys_lines = []
                #for name in self.sys_names:
                #    sys_lines = self.get_sys_lines(name)
                #    sys_lines = [detokenize(ss) for ss in sys_lines]
                #    self.sys_lines.extend(sys_lines)
                #self.idf_hyps = get_idf_dict(self.sys_lines)
                print(f'MoverScore setup finished. Begin calculating MoverScore.')

                start = time.time()
                if not self.multi_ref:
                    ref_lines = self.single_ref_lines
                    ref_lines = [detokenize(ss) for ss in ref_lines]
                    idf_refs = get_idf_dict(ref_lines)
                else:
                    ref_lines = self.multi_ref_lines
                    ref_lines = [[detokenize(ww) for ww in ss] for ss in ref_lines]
                    idf_refs = get_idf_dict(sum(ref_lines, []))
                s_d = {}
                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve}

                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(ss) for ss in sys_lines]
                    if args.hypo_transform is not None:
                        sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, transform_d = transform_d, idx = kk) for kk, line in enumerate(sys_lines)] 
                        sys_lines = [detokenize(ss) for ss in sys_lines]

                    idf_hyps = get_idf_dict(sys_lines) #i calcuate idf here so that it won't be affected the number of system evaluated
                    if not self.multi_ref:
                        scores = word_mover_score(ref_lines, sys_lines, idf_refs, idf_hyps, self.stop_words,
                                                  n_gram=1, remove_subwords=True, batch_size=8, device=self.device)
                    else:
                        scores = np.zeros(len(sys_lines))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_scores = word_mover_score(ref_list, sys_lines, idf_refs, idf_hyps,
                                                           self.stop_words, n_gram=1, remove_subwords=True,
                                                           batch_size=8, device=self.device)
                            scores += np.array(curr_scores)
                        scores = scores / self.ref_num
                    s_d[sys_name] = np.mean(scores)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['mover_score'] = scores[counter]
                        counter += 1
                
                print(f'Finished calculating MoverScore, time passed {time.time() - start}s.')
                s_lis = []
                for sys_name in s_d:
                    logger.info('%s: %f', sys_name, s_d[sys_name])
                    s_lis.append(s_d[sys_name])
                logger.info('avg: %f median: %f', np.mean(s_lis), np.median(s_lis))

            elif metric_name == 'rouge':
                from rouge_score import rouge_scorer
                print(f'Begin calculating ROUGE.')
                start = time.time()
                if not self.multi_ref: #why lower???? rouge is case-insensitive!
                    ref_lines = [detokenize(line) for line in self.single_ref_lines] #[line.lower() for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(line) for line in ww] for ww in self.multi_ref_lines] #[[text.lower() for text in line] for line in self.multi_ref_lines]
                
                src_lines = self.get_src_lines() #source documents
                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve, 'wfreq_d': self.wfreq_d, 'wlogprob_d': self.wlogprob_d}

                rouge_var = ['rouge2', 'rougeL']; s_d = {rv: {} for rv in rouge_var};
                scorer = rouge_scorer.RougeScorer(rouge_var, use_stemmer=True)
                for sys_name in self.sys_names:
                    for rv in rouge_var: s_d[rv][sys_name] = [];
                    sys_lines = self.get_sys_lines(sys_name)

                    if args.hypo_transform is not None:
                        sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, src_line = None, idx = kk, transform_d = transform_d) for kk, line in enumerate(sys_lines)] 
                    sys_lines = [detokenize(line) for line in sys_lines]

                    for i in range(len(sys_lines)):

                        if self.multi_ref:
                            s = scorer.score_multi(ref_lines[i], sys_lines[i])
                        else:
                            s = scorer.score(ref_lines[i], sys_lines[i])
                        
                        for rv in rouge_var:
                            s_d[rv][sys_name].append(s[rv].fmeasure)
                        

                logger.info(f'Finished calculating ROUGE, time passed {time.time() - start}s.')
                for rv in rouge_var:
                    logger.info('summary for %s', rv); tmp_d = {}
                    s_lis = []
                    for sys_name in s_d[rv]:
                        tmp_d[sys_name] = np.mean(s_d[rv][sys_name]);
                        s_lis.append(tmp_d[sys_name])
                    for sn, va in sorted(tmp_d.items(), key = lambda x:x[1], reverse = True):
                        logger.info('%s : %f', sn, va)
                    logger.info('avg: %f median: %f', np.mean(s_lis), np.mean(s_lis))
                    logger.info('')
                breakpoint()

            elif metric_name == 'grouge':
                from gehrmann_rouge_opennmt.rouge_baselines.baseline import baseline_main

                def rouge(dic):
                    """ Get r, p, f scores """
                    r1_, r2_, rl_ = [], [], []
                    for k in dic:
                        r1_.append([dic[k]['rouge_1_recall'], dic[k]['rouge_1_precision'], dic[k]['rouge_1_f_score']])
                        r2_.append([dic[k]['rouge_2_recall'], dic[k]['rouge_2_precision'], dic[k]['rouge_2_f_score']])
                        rl_.append([dic[k]['rouge_l_recall'], dic[k]['rouge_l_precision'], dic[k]['rouge_l_f_score']])
                    return r1_, r2_, rl_

                print(f'Begin calculating ROUGE.')
                start = time.time()
                blockPrint()

                if not self.multi_ref:
                    ref_lines = [line.lower() for line in self.single_ref_lines]
                else:
                    ref_lines = [[text.lower() for text in line] for line in self.multi_ref_lines]

                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [line.lower() for line in sys_lines]

                    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
                    write_list_to_file(sys_lines, 'hypo.txt')
                    if not self.multi_ref:
                        write_list_to_file(ref_lines, 'ref.txt')
                        args = argparse.Namespace(check_repeats=True, delete=True, get_each_score=True, stemming=True,
                                                  method='sent_no_tag', n_bootstrap=1000, run_google_rouge=False,
                                                  run_rouge=True, source='./hypo.txt', target='./ref.txt',
                                                  ref_sep='||NEVER||', num_ref=1, temp_dir='./temp/')

                        scores = baseline_main(args, return_pyrouge_scores=True)['individual_score_results']
                        rouge1_scores, rouge2_scores, rougel_scores = rouge(scores)
                    else:
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            write_list_to_file(ref_list, 'ref.txt')
                            args = argparse.Namespace(check_repeats=True, delete=True, get_each_score=True,
                                                      stemming=True,
                                                      method='sent_no_tag', n_bootstrap=1000, run_google_rouge=False,
                                                      run_rouge=True, source='./hypo.txt', target='./ref.txt',
                                                      ref_sep='||NEVER||', num_ref=1, temp_dir='./temp/')

                            scores = baseline_main(args, return_pyrouge_scores=True)['individual_score_results']
                            r1, r2, rl = rouge(scores)
                            rouge1_scores.append(r1)
                            rouge2_scores.append(r2)
                            rougel_scores.append(rl)
                        rouge1_scores = np.mean(rouge1_scores, axis=0)
                        rouge2_scores = np.mean(rouge2_scores, axis=0)
                        rougel_scores = np.mean(rougel_scores, axis=0)

                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            'rouge1_r': rouge1_scores[counter][0],
                            'rouge1_p': rouge1_scores[counter][1],
                            'rouge1_f': rouge1_scores[counter][2],
                            'rouge2_r': rouge2_scores[counter][0],
                            'rouge2_p': rouge2_scores[counter][1],
                            'rouge2_f': rouge2_scores[counter][2],
                            'rougel_r': rougel_scores[counter][0],
                            'rougel_p': rougel_scores[counter][1],
                            'rougel_f': rougel_scores[counter][2]
                        })
                        counter += 1
                enablePrint()
                os.system('rm -rf hypo.txt ref.txt')
                print(f'Finished calculating ROUGE, time passed {time.time() - start}s.')

            elif metric_name == 'comet' or metric_name == 'cometqe':
                from comet import load_from_checkpoint
                if metric_name == 'comet':
                    model_path = 'models/wmt20-comet-da.save'
                if metric_name == 'cometqe':
                    model_path = 'models/wmt21-comet-qe-mqm.save' #for this the 'ref' won't be used
                print('comet: loading from', model_path)
                model = torch.load(model_path)
                
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]

                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]

                s_d = {}
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    
                    hyp_samples = []
                    for i in range(len(ref_lines)):
                        hyp_samples.append({'src': src_lines[i], 'ref': ref_lines[i], 'mt': sys_lines[i]})
                    print(f'Begin calculating COMET.')
                    scores_seg, score = model.predict(hyp_samples)
                    s_d[sys_name] = score

                s_lis = []
                for sys_name in s_d:
                    logger.info('%s: %f', sys_name, s_d[sys_name])
                    s_lis.append(s_d[sys_name])
                logger.info('avg: %f median: %f std: %f', np.mean(s_lis), np.median(s_lis), np.std(s_lis))

            elif metric_name == 'prism':
                from prism import Prism
                # Set up Prism
                self.prism = Prism(model_dir='./models/m39v1/', lang='en')
                print(f'PRISM setup finished. Begin calculating PRISM.')

                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    # Calculate Both src-based and ref-based
                    src_hypo_scores = self.prism.score(cand=sys_lines, src=src_lines, segment_scores=True)
                    if not self.multi_ref:
                        ref_hypo_scores, hypo_ref_scores, scores = self.prism.score(cand=sys_lines, ref=ref_lines,
                                                                                    segment_scores=True)
                    else:
                        total_num = len(sys_lines)
                        ref_hypo_scores, hypo_ref_scores, scores = np.zeros(total_num), np.zeros(total_num), np.zeros(
                            total_num)
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_ref_hypo_scores, curr_hypo_ref_scores, curr_scores = self.prism.score(cand=sys_lines,
                                                                                                       ref=ref_list,
                                                                                                       segment_scores=True)
                            ref_hypo_scores += curr_ref_hypo_scores
                            hypo_ref_scores += curr_hypo_ref_scores
                            scores += curr_scores

                        ref_hypo_scores = ref_hypo_scores / self.ref_num
                        hypo_ref_scores = hypo_ref_scores / self.ref_num
                        scores = scores / self.ref_num

                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_ref_hypo'] = ref_hypo_scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_hypo_ref'] = hypo_ref_scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_avg'] = scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_src_hypo'] = src_hypo_scores[counter]
                        counter += 1
                print(f'Finished calculating PRISM, time passed {time.time() - start}s.')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                """ Vanilla BARTScore, BARTScore-CNN, BARTScore-CNN-Para """
                from bart_score import BARTScorer

                if args.eval_model_ckpt is not None:
                    logger.info('BART-SCORE will use a loaded model for evaluation: %s', args.eval_model_ckpt)
                    assert('para' not in metric_name)
                    bart_scorer = BARTScorer(device=self.device, model_base = args.eval_model_base, checkpoint = args.eval_model_ckpt)
                else:
                    # Set up BARTScore
                    if 'cnn' in metric_name:
                        bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    elif 'para' in metric_name:
                        logger.info('loading the para model')
                        bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                        bart_scorer.load() #this line loads the paraphrase model
                    else:
                        bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')
                logger.info(f'BARTScore setup finished. Begin calculating BARTScore.')
                if args.hypo_transform is not None: 
                    logger.info('will apply transform %s', args.hypo_transform)
                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines() #source documents
                src_lines = [detokenize(line) for line in src_lines]
                
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]
                
                transform_d = {'srcs': src_lines, 'refs': ref_lines, 'wfreq_d': self.wfreq_d, 'wlogprob_d': self.wlogprob_d} 
                if self.multi_ref:
                    transform_d['refs_reserve'] = self.multi_ref_lines_reserve

                if args.hypo_transform and 'modelgen' in args.hypo_transform: 
                    model_name = args.hypo_transform.split('_')[1]
                    logger.info('getting model generation (transform modelgen) %s', model_name)
                    #torch.save({'src_lines': src_lines, 'ref_lines': ref_lines}, '../plm_ft/saves/summeval_src_lines.save') 
                    if model_name == 'self':
                        model_gens = bart_scorer.get_model_gens(src_lines, batch_size = 4)
                    if model_name == 'load':
                        logger.info('loading model generation from %s', args.modelgen_load)
                        model_gens = []; gens = torch.load(args.modelgen_load);
                        for idx, tt in enumerate(gens):
                            assert(tt['src_line'] == src_lines[idx])
                            model_gens.append(tt['dec_line'])
                    logger.info('model generation complete.')
                    if '_shuffle' in args.hypo_transform:
                        logger.info('shuffling model_gens!') 
                        random.shuffle(model_gens)
                    transform_d['model_gens'] = model_gens
                
                sys_list_done = []
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    if args.hypo_transform is not None:
                        sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, src_line = src_lines[kk], idx = kk, transform_d = transform_d) for kk, line in enumerate(sys_lines)] 
                    src_hypo = bart_scorer.score(src_lines, sys_lines, batch_size=4) #scores of src->hypo
                    if not self.multi_ref: #multi_ref is False
                        ref_hypo = np.array(bart_scorer.score(ref_lines, sys_lines, batch_size=4))
                        hypo_ref = np.array(bart_scorer.score(sys_lines, ref_lines, batch_size=4))
                    else:
                        ref_hypo, hypo_ref = np.zeros(len(sys_lines)), np.zeros(len(sys_lines))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_ref_hypo = np.array(bart_scorer.score(ref_list, sys_lines, batch_size=4))
                            curr_hypo_ref = np.array(bart_scorer.score(sys_lines, ref_list, batch_size=4))
                            ref_hypo += curr_ref_hypo
                            hypo_ref += curr_hypo_ref
                        ref_hypo = ref_hypo / self.ref_num
                        hypo_ref = hypo_ref / self.ref_num
                    avg_f = (ref_hypo + hypo_ref) / 2
                    harm_f = (ref_hypo * hypo_ref) / (ref_hypo + hypo_ref)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            f'{metric_name}_src_hypo': src_hypo[counter],
                            f'{metric_name}_hypo_ref': hypo_ref[counter],
                            f'{metric_name}_ref_hypo': ref_hypo[counter],
                            f'{metric_name}_avg_f': avg_f[counter],
                            f'{metric_name}_harm_f': harm_f[counter]
                        })
                        counter += 1
                    sys_list_done.append(sys_name)
                    arht = args.hypo_transform
                    if arht is not None and (arht.startswith('refreserve') or arht.startswith('highfreqrandom') or arht.startswith('highfreqsource')):
                        logger.info('stopped evaling more system as the transform did not depend on hypothesis')
                        break #in this case just do one system is enough
                logger.info(f'Finished calculating BARTScore, time passed {time.time() - start}s.')
                
                if 'cnn' in metric_name:
                    vs = ['src_hypo']
                if 'para' in metric_name:
                    vs = ['hypo_ref', 'ref_hypo']
                for variant in vs:
                    logger.info('===variant below: %s ===', variant)
                    sys_scores = {}
                    mn = metric_name + '_' + variant
                    for doc_id in self.data:
                        sys_summs = self.data[doc_id]['sys_summs']
                        for sys_name in sys_list_done: #sys_summs:
                            if not sys_name in sys_scores: sys_scores[sys_name] = []
                            sys_scores[sys_name].append(sys_summs[sys_name]['scores'][mn])

                    avg_scores, best_score, best_sys_name = [], -10000, 'NONE'
                    for sys_name in sys_list_done: #sys_scores:
                        avg_s = np.mean(sys_scores[sys_name])
                        logger.info('%s len: %d avg: %f', sys_name, len(sys_scores[sys_name]), avg_s)
                        avg_scores.append(avg_s)
                        if avg_s > best_score:
                            best_score, best_sys_name = avg_s, sys_name
                    logger.info('===avg scores for %s: %f median: %f', mn, np.mean(avg_scores), np.median(avg_scores))
                    logger.info('===best_socre: %f best_sys: %s ===', best_score, best_sys_name)
             
            elif metric_name.startswith('prompt'):
                """ BARTScore adding prompts """
                from bart_score import BARTScorer
                logger.info('prompt metric %s', metric_name)

                def prefix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(p + ', ' + x)
                    return new_l

                def suffix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(x + ' ' + p + ',')
                    return new_l

                if 'cnn' in metric_name:
                    name = 'bart_score_cnn'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    name = 'bart_score_para'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    name = 'bart_score'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                logger.info(f'BARTScore-P setup finished. Begin calculating BARTScore-P.')
                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]

                # SRC -> HYPO prompt
                if 'src' in metric_name:
                    logger.info('prompt num: %d', len(SRC_HYPO))
                    for prompt in SRC_HYPO:
                        logger.info('doing prompt: %s', prompt)
                        for sys_name in self.sys_names:
                            sys_lines = self.get_sys_lines(sys_name)
                            sys_lines = [detokenize(line) for line in sys_lines]
                            src_hypo_en = bart_scorer.score(suffix_prompt(src_lines, prompt), sys_lines, batch_size=4)
                            src_hypo_de = bart_scorer.score(src_lines, prefix_prompt(sys_lines, prompt), batch_size=4)
                            counter = 0
                            for doc_id in self.data:
                                self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                                    f'{name}_src_hypo_en_{prompt}': src_hypo_en[counter],
                                    f'{name}_src_hypo_de_{prompt}': src_hypo_de[counter]
                                })
                                counter += 1

                # REF <-> HYPO prompt
                if 'ref' in metric_name:
                    for prompt in REF_HYPO:
                        for sys_name in self.sys_names:
                            sys_lines = self.get_sys_lines(sys_name)
                            sys_lines = [detokenize(line) for line in sys_lines]
                            if not self.multi_ref:
                                ref_hypo_en = np.array(
                                    bart_scorer.score(suffix_prompt(ref_lines, prompt), sys_lines, batch_size=4))
                                hypo_ref_en = np.array(
                                    bart_scorer.score(suffix_prompt(sys_lines, prompt), ref_lines, batch_size=4))
                                ref_hypo_de = np.array(
                                    bart_scorer.score(ref_lines, prefix_prompt(sys_lines, prompt), batch_size=4))
                                hypo_ref_de = np.array(
                                    bart_scorer.score(sys_lines, prefix_prompt(ref_lines, prompt), batch_size=4))
                            else:
                                ref_hypo_en, hypo_ref_en, ref_hypo_de, hypo_ref_de = np.zeros(len(sys_lines)), np.zeros(
                                    len(sys_lines)), \
                                                                                     np.zeros(len(sys_lines)), np.zeros(
                                    len(sys_lines))
                                for i in range(self.ref_num):
                                    ref_list = [x[i] for x in ref_lines]
                                    curr_ref_hypo_en = np.array(
                                        bart_scorer.score(suffix_prompt(ref_list, prompt), sys_lines, batch_size=4))
                                    curr_hypo_ref_en = np.array(
                                        bart_scorer.score(suffix_prompt(sys_lines, prompt), ref_list, batch_size=4))
                                    curr_ref_hypo_de = np.array(
                                        bart_scorer.score(ref_list, prefix_prompt(sys_lines, prompt), batch_size=4))
                                    curr_hypo_ref_de = np.array(
                                        bart_scorer.score(sys_lines, prefix_prompt(ref_list, prompt), batch_size=4))
                                    ref_hypo_en += curr_ref_hypo_en
                                    hypo_ref_en += curr_hypo_ref_en
                                    ref_hypo_de += curr_ref_hypo_de
                                    hypo_ref_de += curr_hypo_ref_de
                                ref_hypo_en = ref_hypo_en / self.ref_num
                                hypo_ref_en = hypo_ref_en / self.ref_num
                                ref_hypo_de = ref_hypo_de / self.ref_num
                                hypo_ref_de = hypo_ref_de / self.ref_num
                            avg_f_en = (ref_hypo_en + hypo_ref_en) / 2
                            avg_f_de = (ref_hypo_de + hypo_ref_de) / 2
                            harm_f_en = (ref_hypo_en * hypo_ref_en) / (ref_hypo_en + hypo_ref_en)
                            harm_f_de = (ref_hypo_de * hypo_ref_de) / (ref_hypo_de + hypo_ref_de)
                            counter = 0
                            for doc_id in self.data:
                                self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                                    f'{name}_hypo_ref_en_{prompt}': hypo_ref_en[counter],
                                    f'{name}_ref_hypo_en_{prompt}': ref_hypo_en[counter],
                                    f'{name}_avg_f_en_{prompt}': avg_f_en[counter],
                                    f'{name}_harm_f_en_{prompt}': harm_f_en[counter],
                                    f'{name}_hypo_ref_de_{prompt}': hypo_ref_de[counter],
                                    f'{name}_ref_hypo_de_{prompt}': ref_hypo_de[counter],
                                    f'{name}_avg_f_de_{prompt}': avg_f_de[counter],
                                    f'{name}_harm_f_de_{prompt}': harm_f_de[counter]
                                })
                                counter += 1
                logger.info(f'Finished calculating BARTScore-P, time passed {time.time() - start}s.')

            else:
                raise NotImplementedError

    def select_multi(self, metric_name):
        logger.info('SELECT_MULTI! from multi_refs metric_name: %s', metric_name)
        if metric_name == 'bert_score':
            from bert_score import BERTScorer

            # Set up BERTScore
            bert_scorer = BERTScorer(
                lang='en',
                idf=False,
                rescale_with_baseline=True,
                device=self.device
            )
            logger.info(f'BERTScore setup finished. Begin calculating BERTScore.')

        if metric_name.startswith('comet'):
            from comet import load_from_checkpoint
            if metric_name == 'comet':
                model_path = 'models/wmt20-comet-da.save'
            if metric_name == 'cometqe':
                model_path = 'models/wmt21-comet-qe-mqm.save' #for this the 'ref' won't be used
            print('comet: loading from', model_path)
            model = torch.load(model_path)

        if metric_name.startswith('bart_score'):
            from bart_score import BARTScorer
            if 'cnn' in metric_name:
                bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
            elif 'para' in metric_name:
                bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                bart_scorer.load() #this line loads the paraphrase model            

        if metric_name == 'mover_score':
            from moverscore import word_mover_score, get_idf_dict
            # Set up MoverScore
            with open('files/stopwords.txt', 'r', encoding='utf-8') as f:
                self.stop_words = set(f.read().strip().split(' '))

        if metric_name == 'rouge':
            from rouge_score import rouge_scorer
            logger.info(f'Begin calculating ROUGE.')
            rouge_var = 'rougeL'; s_d = {rv: {} for rv in rouge_var};
            logger.info('rouge_var: %s', str(rouge_var))
            scorer = rouge_scorer.RougeScorer([rouge_var], use_stemmer=True)

        start = time.time()
        assert(self.multi_ref == True)
        multi_refs = self.single_ref_lines if not self.multi_ref else self.multi_ref_lines
        multi_refs = [[detokenize(line) for line in ww] for ww in multi_refs]

        #multi_refs = self.multi_ref_lines_reserve
        src_lines = self.get_src_lines() #source documents
        src_lines = [detokenize(line) for line in src_lines]
        
        best_scores, best_idx, all_scores = [-1000 for kk in src_lines], [0 for kk in src_lines], [[] for kk in src_lines]
        for idx in range(len(multi_refs[0])):
            sys_lines = [rr[idx] for rr in multi_refs]
            ref_lines = [rr[:idx] + rr[idx + 1:] for rr in multi_refs]
            #sys_lines = [detokenize(line) for line in sys_lines]
            
            if metric_name.startswith('comet'):
                hyp_samples = []
                for i in range(len(ref_lines)):
                    hyp_samples.append({'src': src_lines[i], 'ref': ref_lines[i], 'mt': sys_lines[i]})
                print(f'Begin calculating COMET.')
                scores_seg, score = model.predict(hyp_samples)
                s_now = scores_seg

            if metric_name == 'mover_score':
                idf_refs = get_idf_dict(ref_lines)
                idf_hyps = get_idf_dict(sys_lines)
                s_now = word_mover_score(ref_lines, sys_lines, idf_refs, idf_hyps, self.stop_words, n_gram=1, remove_subwords=True, batch_size=8, device=self.device)

            if metric_name == 'bert_score':
                P, R, F = bert_scorer.score(sys_lines, ref_lines)
                s_now = F.tolist()

            if metric_name == 'rouge':
                s_now = []
                for i in range(len(sys_lines)):
                    s = scorer.score_multi(ref_lines[i], sys_lines[i])
                    s_now.append(s[rouge_var].fmeasure)

            if metric_name.startswith('bart_score'):
                src_hypo = bart_scorer.score(src_lines, sys_lines, batch_size=4)
                s_now = src_hypo

            for i in range(len(ref_lines)):
                all_scores[i].append(s_now[i])
                if s_now[i] > best_scores[i]:
                    best_scores[i] = s_now[i]
                    best_idx[i] = idx
        
        select_refs = [multi_refs[i][best_idx[i]] for i in range(len(ref_lines))]
        select_multi_refs = [(multi_refs[i][:best_idx[i]] + multi_refs[i][best_idx[i]+1:]) for i in range(len(ref_lines))]
        save_d = {'select_refs': select_refs, 'select_multi_refs': select_multi_refs}
        logger.info('score now: %f', np.mean(best_scores))
        save_fn = f'selectref_evalwithmultiref/best_ref_{metric_name}.save'
        logger.info('saving best_refs to %s, will wait 5 seconds', save_fn); time.sleep(6);
        torch.save(save_d, save_fn)
        breakpoint()
        x=1+1


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--multi_ref', action='store_true', default=False,
                        help='Whether we are using multiple references to calculate scores.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--mover_score', action='store_true', default=False,
                        help='Whether to calculate MoverScore')
    parser.add_argument('--rouge', action='store_true', default=False,
                        help='Whether to calculate ROUGE')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Whether to calculate BARTScore-P. Can be bart_src, bart_ref, bart_cnn_src, '
                             'bart_cnn_ref, bart_para_src, bart_para_ref')
    
    parser.add_argument('--hypo_transform', type=str, default=None, help='transform the hypo (system output) for sanity check purposes')
    parser.add_argument('--modelgen_load', type=str, default=None, help = 'the place to load model generation')
    parser.add_argument('--select_multi', action='store_true', default=False)
    parser.add_argument('--select_multi_metric', type=str, default=None)
    parser.add_argument('--use_select_multi', action='store_true', default=False)
    
    parser.add_argument('--eval_model_ckpt', type=str, default=None, help='use a model different from facebook/bart-cnn-large')
    parser.add_argument('--eval_model_base', type=str, default=None, help='use a model different from facebook/bart-cnn-large')

    args = parser.parse_args()

    scorer = Scorer(args, args.file, args.device, args.multi_ref)

    if args.select_multi:
        scorer.select_multi(args.select_multi_metric)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.mover_score:
        METRICS.append('mover_score')
    if args.rouge:
        METRICS.append('rouge')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prism:
        METRICS.append('prism')
    if args.comet:
        METRICS.append('comet')
    if args.prompt is not None:
        prompt = args.prompt
        assert prompt in ['bart_src', 'bart_ref', 'bart_cnn_src',
                          'bart_cnn_ref', 'bart_para_src', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()
