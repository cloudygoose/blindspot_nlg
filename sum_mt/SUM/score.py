import argparse, itertools
import os, sys, torch, time
from pathlib import Path
import time, random, json, math, editdistance
import numpy as np
from utils import *
from datasets import load_dataset
import random
from nltk import word_tokenize, sent_tokenize

#sys.path.append('/home/gridsan/tianxing/txml_shared/projects/metricnlg_2205/BARTScore')
sys.path.append(str(Path(__file__).absolute().parent.parent))
import sanity_transform
import score_utils
import transform_utils

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(); logger.handlers = [];
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

SRC_HYPO = read_file_to_list('files/src_hypo_prompt.txt')
REF_HYPO = read_file_to_list('files/ref_hypo_prompt.txt')

def report_stat(s_lis):
    logger.info('upper quartile: %f', np.quantile(s_lis, 0.75))
    logger.info('best: %f avg: %f median: %f std: %f', np.max(s_lis), np.mean(s_lis), np.median(s_lis), np.std(s_lis))

def reduce_mean(args, res_d, info_d): #merge the results of different random seed
    for mn in res_d:
        if args.hypo_transform is not None:
            for hypo_transform in args.hypo_transform.split(','):
                if hypo_transform.endswith('[seed]'):
                    seed_lis = [1,2,3,4,5]
                    hypo_transform = hypo_transform[:-6]
                else:
                    seed_lis = [1]
                lis_now = []; lis_edit = [];
                for seed_now in seed_lis:
                    sn = 'ref' + '-' + hypo_transform + f'-seed{seed_now}'
                    assert(sn in res_d[mn])
                    lis_now.append(res_d[mn][sn]); lis_edit.append(info_d[sn]['edit_ratio'])
                    del res_d[mn][sn]
                reduce_sn = 'ref' + '_' + hypo_transform + ('_seedreduce' if len(seed_lis) > 1 else '')
                logger.info('%s reducing randomseed results to %s , lis_now: %s', mn, reduce_sn, str(lis_now))
                res_d[mn][reduce_sn] = {'mean': np.mean(lis_now), 'edit_ratio': np.mean(lis_edit), 'std': np.std(lis_now)}
        res_d[mn]['ref'] = {'mean': res_d[mn]['ref'], 'edit_ratio': 0, 'std': 0}
    return res_d

class Scorer:
    """ Support ROUGE-1,2,L, BERTScore, MoverScore, PRISM, BARTScore """

    def __init__(self, args, file_path, device='cuda:0', multi_ref=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including src, ref_summ, ref_summs, and sys_summ.
        """
        self.multi_ref = multi_ref
        self.device = device
        self.data = read_pickle(file_path)
        logger.info(f'Data loaded from {file_path}.')
        logger.info(f'Overwriting with CNNDM data... as they are cleaner')
        self.load_from_cnndm_ori()
        logger.info(f'detokenzing every system outputs...')
        logger.info(f'remember to use nltk_detokenize')
        for idx, doc_id in enumerate(self.data):
            for sn in self.data[doc_id]['sys_summs']:
                self.data[doc_id]['sys_summs'][sn]['sys_summ'] = nltk_detokenize(self.data[doc_id]['sys_summs'][sn]['sys_summ'])

        self.args = args

        self.multi_ref_lines, self.multi_ref_lines_reserve = self.get_multi_ref_lines()
        self.srcs = self.get_src_lines()
        self.sys_names = self.get_sys_names()
        #self.refs = self.get_single_ref_lines() #changed to single_ref_lines, to be clear


        """ #this block is originally used for select_ref
        if args.multi_ref:
            #load_fn = 'select_multiref/best_ref_comet.save'
            load_fn = 'selectref_evalwithmultiref/best_ref_rouge.save'
            #load_fn = 'select_multiref/best_ref_rouge.save'
            #load_fn = 'select_multiref/best_ref_bert_score.save'
            #load_fn = 'select_multiref/best_ref_bart_score_cnn.save'
            #load_fn = 'select_multiref/best_ref_mover_score.save'
            ld = torch.load(load_fn)
            select_refs, select_multi_refs = ld['select_refs'], ld['select_multi_refs']

            if args.use_select_multi:
                logger.info('loading human gen from %s', load_fn); time.sleep(2)
                self.sys_names = ['human', 'M0']
                for idx, doc_id in enumerate(self.data):
                    self.data[doc_id]['sys_summs']['human'] = {'sys_summ': select_refs[idx], 'scores': {}}
            
            logger.info('also loading multi_refs'); time.sleep(2)
            self.multi_ref_lines, self.multi_ref_lines_reserve = select_multi_refs, [] #in this case we shouldn't be using reserve
        #else:
        #    self.sys_names = self.get_sys_names()
        """

        self.single_ref_lines = self.get_single_ref_lines()

        if not multi_ref:
            self.ref_lines = self.single_ref_lines
            print(f'In a single-reference setting.')
        else:
            self.ref_num = len(self.multi_ref_lines[0])
            print(f'In a multi-reference setting.')
            self.ref_lines = self.multi_ref_lines

        if args.all_sys:
            self.sys_names.append('ref')
        else:
            self.sys_names = ['ref'] #if you want to run all systems, comment out this line
        for idx, doc_id in enumerate(self.data):
            self.data[doc_id]['sys_summs']['ref'] = {'sys_summ': self.single_ref_lines[idx], 'scores': {}}

        ld_fn = '../plm_ft/cnndm_saves/wfreq.save'
        logger.info('loading wfreq from %s', ld_fn)
        freq_d = torch.load(ld_fn)
        f_sum = sum([w[1] for w in freq_d.items()])
        logprob_d = {w: math.log(v * 1.0 / f_sum) for w,v in freq_d.items()}
        self.wfreq_d, self.wlogprob_d = freq_d, logprob_d

        transform_d = {'src': self.srcs, 'refs': self.single_ref_lines, 'wfreq_d': self.wfreq_d, 'wlogprob_d': self.wlogprob_d} 
        transform_d['all_sents'] = list(itertools.chain(*[sent_tokenize(ss) for ss in self.single_ref_lines])); assert(len(transform_d['all_sents']) > len(self.single_ref_lines))

        if args.hypo_transform and ('freq3gram' in args.hypo_transform or 'freq4gram' in args.hypo_transform):
            logger.info('computing freq ngram from ../plm_ft/cnndm_saves/train_summaries.save')
            train_summaries = torch.load('../plm_ft/cnndm_saves/train_summaries.save')
            ngram_freqd = transform_utils.compute_ngram_freq(train_summaries)
            transform_d['ngram_freqd'] = ngram_freqd 

        spacytrf_save_fn = './spacy_saves/' + args.file.replace('/', '.') + '.single_ref.spacy_trf_save'
        if args.save_spacy_trf:
            #save to ./spacy_saves/
            lines = self.single_ref_lines;
            nlp_trf = sanity_transform.load_spacy_trf()
            spacy_d = {}; logger.info('processing spacy_trf to lines')
            for line in lines: 
                spacy_d[line] = nlp_trf(line)
                for sent in sent_tokenize(line):
                    spacy_d[sent] = nlp_trf(sent)
            logger.info('saving to %s, exiting...', spacytrf_save_fn); torch.save(spacy_d, spacytrf_save_fn)
            sys.exit(0)
        else:
            logger.info('loading spacy_trf processed from %s', spacytrf_save_fn)
            transform_d['spacy_trf_d'] = torch.load(spacytrf_save_fn)

        if args.hypo_transform and 'modelgen' in args.hypo_transform: 
            #logger.info('this part of the code needs to be moved forward')
            #sys.exit(1)
            model_name = args.hypo_transform.split('_')[1]
            logger.info('getting model generation (transform modelgen) %s', model_name)
            #torch.save({'src_lines': src_lines, 'ref_lines': ref_lines}, '../plm_ft/saves/summeval_src_lines.save') 
            if model_name == 'self':
                model_gens = bart_scorer.get_model_gens(src_lines, batch_size = 4)
            if model_name == 'load':
                logger.info('loading model generation from %s', args.modelgen_load)
                model_gens = []; gens = torch.load(args.modelgen_load);
                for idx, tt in enumerate(gens):
                    #print(editdistance.eval(tt['src_line'].split(), self.srcs[idx].split()), len(self.srcs[idx].split()))
                    #assert(editdistance.eval(tt['src_line'].split(), self.srcs[idx].split()) < 100)
                    assert(tt['src_line'] == self.srcs[idx]) #src_lines[idx]) #we did some further preprocess, causing the src line to be a little different
                    model_gens.append(tt['dec_line'])
            logger.info('model generation complete.')
            if '_shuffle' in args.hypo_transform:
                logger.info('shuffling model_gens!') 
                random.shuffle(model_gens)
            transform_d['model_gens'] = model_gens

        self.transform_info = {}
        if args.hypo_transform is not None:
            for hypo_transform in args.hypo_transform.split(','):
                seed_lis = [1]
                if hypo_transform.endswith('[seed]'):
                    seed_lis = [1,2,3,4,5]
                    hypo_transform = hypo_transform[:-6]
                
                for seed_now in seed_lis:
                    sn = 'ref' + '-' + hypo_transform + f'-seed{seed_now}'; self.transform_info[sn] = {}
                    #random.seed(seed_now); np.random.seed(seed_now);
                    logger.info('applying transform as %s', sn)
                    tf_lines, tf_stat = sanity_transform.batch_sanity_transform(args, self.single_ref_lines, hypo_transform, transform_d, seed = seed_now)
                    logger.info('applied transform as %s edit_ratio(percentage): %f change_ratio(percentage): %f', sn, tf_stat['edit_ratio'] * 100, tf_stat['change_ratio'] * 100) 
                    self.transform_info[sn]['edit_ratio'] = tf_stat['edit_ratio']
                    for kk, doc_id in enumerate(self.data):
                        self.data[doc_id]['sys_summs'][sn] = {'sys_summ': tf_lines[kk], 'scores': {}}
                    self.sys_names.append(sn)

    def get_sys_names(self):
        first_id = list(self.data.keys())[0]
        return list(self.data[first_id]['sys_summs'].keys())

    def get_single_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summ'])
        #ref_lines = [detokenize(ww) for ww in ref_lines]

        return ref_lines

    def load_from_cnndm_ori(self):
        test_fn =  './files/cnndm_saves/test.json' #'../plm_ft/cnndm_saves/test.json'
        data_files = {
            'test': test_fn,
        }
        extension = test_fn.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files); 
        data_d = {}
        for sample in raw_datasets['test']:
            data_d[sample['id']] = sample
        for doc_id in self.data:
            idx = doc_id.split('-')[-1]
            assert(idx in data_d)
            self.data[doc_id]['ref_summ'] = data_d[idx]['summary']
            self.data[doc_id]['src'] = data_d[idx]['text']
            v = 1+1

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
        logger.info('IMPORTANT: using the last %d as reference!', R_NUM)
        for doc_id in self.data:
        #for i in range(len(ref_lines)):
            rr = mul_d[doc_id.split('-')[2]] #the multi-ref does not have serious tokenization problem, and i can just use it as is.
            #rr = [detokenize(ww) for ww in rr]
            #ref_lines.append(rr[10 - R_NUM:])
            ref_lines_reserve.append(rr[:10 - R_NUM])
            ref_lines.append(rr[(10 - R_NUM):])
            assert(len(ref_lines[-1]) == R_NUM and len(ref_lines_reserve[-1]) == 10 - R_NUM)

        return ref_lines, ref_lines_reserve 

    def get_sys_lines(self, sys_name):
        sys_lines = []
        for doc_id in self.data:
            sys_lines.append(self.data[doc_id]['sys_summs'][sys_name]['sys_summ'])
        #sys_lines = [detokenize(ww) for ww in sys_lines]
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
        res_d = {} #the dict to save all results 
        """ metrics: list of metrics """
        for metric_name in metrics:
            if metric_name == 'bert_score':
                from bert_score import BERTScorer

                # Set up BERTScore
                bert_scorer = BERTScorer(
                    lang='en',
                    idf=False,
                    rescale_with_baseline=True, #this rescale could make the difference larger and more visible
                    device=self.device
                )
                logger.info(f'BERTScore setup finished. Begin calculating BERTScore.')
                if args.hypo_transform is not None: 
                    logger.info('will apply transform %s', args.hypo_transform)
                start = time.time()
                ref_lines = self.single_ref_lines if not self.multi_ref else self.multi_ref_lines

                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve}
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    #if args.hypo_transform is not None: #moved forward
                    #    sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, transform_d = transform_d, idx = kk) for kk, line in enumerate(sys_lines)] 

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
                for variant in ['f', 'p', 'r']: #['r', 'p', 'f']:
                    sys_scores = {}
                    mn = 'bert_score' + '_' + variant
                    for doc_id in self.data:
                        sys_summs = self.data[doc_id]['sys_summs']
                        for sys_name in self.sys_names:
                            if not sys_name in sys_scores: sys_scores[sys_name] = []
                            sys_scores[sys_name].append(sys_summs[sys_name]['scores'][mn])

                    logger.info('below is variant %s', mn); res_d[mn] = {}
                    s_lis = []
                    for sys_name in sys_scores:
                        avg_s = np.mean(sys_scores[sys_name])
                        logger.info('%s len: %d avg: %f', sys_name, len(sys_scores[sys_name]), avg_s)
                        s_lis.append(avg_s)
                        res_d[mn][sys_name] = avg_s
                    report_stat(s_lis)

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
                    idf_refs = get_idf_dict(ref_lines)
                else:
                    ref_lines = self.multi_ref_lines
                    idf_refs = get_idf_dict(sum(ref_lines, []))
                s_d = {}
                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve}

                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    #if args.hypo_transform is not None: #moved forward
                    #    sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, transform_d = transform_d, idx = kk) for kk, line in enumerate(sys_lines)] 
                    #    sys_lines = [detokenize(ss) for ss in sys_lines]

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
                s_lis = []; res_d[metric_name] = {}
                for sys_name in s_d:
                    logger.info('%s: %f', sys_name, s_d[sys_name])
                    s_lis.append(s_d[sys_name]); res_d[metric_name][sys_name] = s_d[sys_name]
                report_stat(s_lis)

            elif metric_name == 'rouge':
                from rouge_score import rouge_scorer
                print(f'Begin calculating ROUGE.')
                start = time.time()
                if not self.multi_ref: #why lower???? rouge is case-insensitive!
                    ref_lines = self.single_ref_lines
                else:
                    ref_lines = self.multi_ref_lines
                
                src_lines = self.get_src_lines() #source documents
                transform_d = {'refs': self.single_ref_lines, 'refs_reserve': self.multi_ref_lines_reserve, 'wfreq_d': self.wfreq_d, 'wlogprob_d': self.wlogprob_d}

                #here i only report f-measure since it's mostly used
                rouge_var = ['rouge2', 'rougeL']; fr_var = ['rouge2-f', 'rougeL-f']; s_d = {rv: {} for rv in fr_var};
                scorer = rouge_scorer.RougeScorer(rouge_var, use_stemmer=True)
                for sys_name in self.sys_names:
                    for rv in fr_var:  s_d[rv][sys_name] = [];
                    sys_lines = self.get_sys_lines(sys_name)

                    #if args.hypo_transform is not None:
                    #    sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, src_line = None, idx = kk, transform_d = transform_d) for kk, line in enumerate(sys_lines)] 
                    #sys_lines = [detokenize(line) for line in sys_lines]

                    for i in range(len(sys_lines)):
                        if self.multi_ref:
                            s = scorer.score_multi(ref_lines[i], sys_lines[i])
                        else:
                            s = scorer.score(ref_lines[i][0], sys_lines[i])
                        
                        for rv in rouge_var:
                            s_d[rv + '-f'][sys_name].append(s[rv].fmeasure)
                            #s_d[rv + '-r'][sys_name].append(s[rv].recall)
                        

                logger.info(f'Finished calculating ROUGE, time passed {time.time() - start}s.')
                for rv in fr_var:
                    #logger.info('summary for %s', rv); 
                    res_d[rv] = {}
                    for sys_name in s_d[rv]:
                        res_d[rv][sys_name] = np.mean(s_d[rv][sys_name]);

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
                    ref_lines = self.sinlge_ref_lines #[detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = self.multi_ref_lines #[[detokenize(text) for text in ww] for ww in self.multi_ref_lines]

                src_lines = self.get_src_lines()

                s_d = {}
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    if not self.multi_ref:
                        hyp_samples = []
                        for i in range(len(ref_lines)):
                            hyp_samples.append({'src': src_lines[i], 'ref': ref_lines[i], 'mt': sys_lines[i]})
                        print(f'Begin calculating COMET.')
                        scores_seg, score = model.predict(hyp_samples)
                    else:
                        s_lis = []
                        for r_idx in range(len(ref_lines[0])):
                            hyp_samples = []
                            for i in range(len(ref_lines)):
                                hyp_samples.append({'src': src_lines[i], 'ref': ref_lines[i][r_idx], 'mt': sys_lines[i]})
                            logger.info(f'Begin calculating COMET. r_idx: %d sys_name: %s', r_idx, sys_name)
                            scores_seg, score = model.predict(hyp_samples)
                            s_lis.append(score)
                        score = np.mean(s_lis)
                    s_d[sys_name] = score

                s_lis = []; res_d[metric_name] = {}
                for sys_name in s_d:
                    logger.info('%s: %f', sys_name, s_d[sys_name])
                    s_lis.append(s_d[sys_name]); res_d[metric_name][sys_name] = s_d[sys_name];
                report_stat(s_lis)

            elif metric_name == 'prism':
                from prism import Prism
                # Set up Prism
                self.prism = Prism(model_dir='./models/m39v1/', lang='en')
                print(f'PRISM setup finished. Begin calculating PRISM.')

                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                if not self.multi_ref:
                    ref_lines = self.single_ref_lines #[detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = self.multi_ref_lines #[[detokenize(text) for text in line] for line in self.multi_ref_lines]
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
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
            
            elif metric_name == 'unieval':
                from unieval_metric.evaluator import get_evaluator
                from unieval_metric.utils import convert_to_json
                # Set up UniEval

                evaluator = get_evaluator('summarization')

                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                if not self.multi_ref:
                    ref_lines = self.single_ref_lines 
                else:
                    ref_lines = self.multi_ref_lines 
                
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    if not self.multi_ref:
                        sys.exit(1) #this should not happen since we currently only work on summEval
                    else:
                        total_num = len(sys_lines)
                        ss_lis = None
                        for refi in range(self.ref_num):
                            ref_list = [x[refi] for x in ref_lines]
                            data = convert_to_json(output_list = sys_lines, src_list = src_lines, ref_list = ref_list)
                            eval_scores = evaluator.evaluate(data, print_result=True)
                            if ss_lis is None:
                                ss_lis = [{kk : [vv] for kk, vv in s.items()} for s in eval_scores]
                            else:
                                for i in range(len(eval_scores)):
                                    for f in eval_scores[i]:
                                        ss_lis[i][f].append(eval_scores[i][f])
                        ss_lis = [{kk: np.mean(vv) for kk,vv in s.items()} for s in ss_lis]

                    s_d, counter = {}, 0
                    for doc_id in self.data:
                        for f in ss_lis[counter]:
                            self.data[doc_id]['sys_summs'][sys_name]['scores']['unieval_{}'.format(f)] = ss_lis[counter][f]
                            if f not in s_d: 
                                s_d[f] = []
                            else:
                                s_d[f].append(ss_lis[counter][f])
                        #self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_ref_hypo'] = ref_hypo_scores[counter]
                        counter += 1
                    
                    for f in s_d:
                        mn = 'unieval_{}'.format(f)
                        if not mn in res_d: res_d[mn] = {}                     
                        res_d[mn][sys_name] = np.mean(s_d[f]);
                
                print(f'Finished calculating UniEval, time passed {time.time() - start}s.')

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
                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines() #source documents
                ref_lines = self.ref_lines

                #if self.multi_ref:
                #    transform_d['refs_reserve'] = self.multi_ref_lines_reserve
                
                sys_list_done = []
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
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
                logger.info(f'Finished calculating BARTScore, time passed {time.time() - start}s.')
                
                if 'cnn' in metric_name:
                    vs = ['hypo_ref', 'ref_hypo', 'src_hypo', 'avg_f']
                if 'para' in metric_name:
                    vs = ['hypo_ref', 'ref_hypo', 'src_hypo', 'avg_f']
                for variant in vs:
                    logger.info('===variant below: %s ===', variant)
                    sys_scores = {}
                    mn = metric_name + '_' + variant; res_d[mn] = {}
                    for doc_id in self.data:
                        sys_summs = self.data[doc_id]['sys_summs']
                        for sys_name in sys_list_done: #sys_summs:
                            if not sys_name in sys_scores: sys_scores[sys_name] = []
                            sys_scores[sys_name].append(sys_summs[sys_name]['scores'][mn])

                    avg_scores, best_score, best_sys_name = [], -10000, 'NONE'
                    for sys_name in sys_list_done: #sys_scores:
                        avg_s = np.mean(sys_scores[sys_name])
                        logger.info('%s len: %d avg: %f', sys_name, len(sys_scores[sys_name]), avg_s)
                        res_d[mn][sys_name] = avg_s
                        avg_scores.append(avg_s)
                        if avg_s > best_score:
                            best_score, best_sys_name = avg_s, sys_name
                    #logger.info('===avg scores for %s: %f median: %f', mn, np.mean(avg_scores), np.median(avg_scores))
                    report_stat(avg_scores)
                    logger.info('===best_score: %f best_sys: %s ===', best_score, best_sys_name)
            
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
                if not self.multi_ref:
                    ref_lines = self.sinlge_ref_lines #[detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = self.multi_ref_lines #[[detokenize(text) for text in line] for line in self.multi_ref_lines]

                # SRC -> HYPO prompt
                if 'src' in metric_name:
                    logger.info('prompt num: %d', len(SRC_HYPO))
                    for prompt in SRC_HYPO:
                        logger.info('doing prompt: %s', prompt)
                        for sys_name in self.sys_names:
                            sys_lines = self.get_sys_lines(sys_name)
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
        
        return res_d


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, default = None, required=False,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--multi_ref', action='store_true', default=False,
                        help='Whether we are using multiple references to calculate scores.')
    parser.add_argument('--output', type=str, default = None, required=False,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--mover_score', action='store_true', default=False,
                        help='Whether to calculate MoverScore')
    parser.add_argument('--rouge', action='store_true', default=False,
                        help='Whether to calculate ROUGE')
    parser.add_argument('--grouge', action='store_true', default=False,
                        help='Whether to calculate GROUGE')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--cometqe', action='store_true')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--unieval', action='store_true')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Whether to calculate BARTScore-P. Can be bart_src, bart_ref, bart_cnn_src, '
                             'bart_cnn_ref, bart_para_src, bart_para_ref')
    
    parser.add_argument('--all_sys', action='store_true') #whether to run all systems, instead of just the ref
    parser.add_argument('--hypo_transform', type=str, default=None, help='transform the hypo (system output) for sanity check purposes')
    parser.add_argument('--debug_transform', action = 'store_true', default=False)

    parser.add_argument('--modelgen_load', type=str, default=None, help = 'the place to load model generation')
    #parser.add_argument('--select_multi', action='store_true', default=False) #moved to legacy
    #parser.add_argument('--select_multi_metric', type=str, default=None) #moved to legacy
    parser.add_argument('--use_select_multi', action='store_true', default=False)
    
    parser.add_argument('--eval_model_ckpt', type=str, default=None, help='use a model different from facebook/bart-cnn-large')
    parser.add_argument('--eval_model_base', type=str, default=None, help='use a model different from facebook/bart-cnn-large')
    parser.add_argument('--save_spacy_trf', action = 'store_true', default=False) #save results from spacy_trf, so that afterwards we can just load it
    parser.add_argument('--save_score', action='store_true', default=False)
    parser.add_argument('--cache_hypo_transform', action='store_true', default=False)

    args = parser.parse_args()
    random.seed(1)

    if args.file is None: args.file = 'SummEval/data.pkl';
    assert(args.file == 'SummEval/data.pkl')
    args.task_name = 'sum'
    logger.info('Assuming working on SummEval, setting args.multi_ref to True...')
    args.multi_ref = True

    if args.hypo_transform is not None:
        ht = args.hypo_transform; 
        if not ht.endswith(','): ht = ht + ','
        if ht == 'flu-all,':
            #ht = (flu-lemmatizeverb,flu-removepreposition-0.7,flu-removestopwords-0.25,flu-noisepunct,' +
            #    'randomworddrop-0.10,flu-randomlocalswap-0.10,flu-randomtokenrep-0.10,flu-sentencemiddleswap,flu-removearticle')
            ht = ('flu-truncate,flu-randomworddrop,flu-randomlocalswap,flu-randomtokenrep,flu-sentencemiddleswap,flu-lemmatizeverb,flu-removepreposition,' + 
                'flu-noisepunct,flu-removestopwords,flu-removearticle,')
            logger.info('flu-all detected, setting hypo_transform to %s', ht)
        ht = ht.replace('flu-truncate,', ''.join(['flu-truncate-A,'.replace('A', str(p)) for p in [0.10, 0.20, 0.30, 0.40, 0.50]]))
        ht = ht.replace('flu-randomworddrop,', ''.join(['flu-randomworddrop-A[seed],'.replace('A', str(p)) for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]))
        ht = ht.replace('flu-randomlocalswap,', ''.join(['flu-randomlocalswap-A[seed],'.replace('A', str(p)) for p in [0.05, 0.15, 0.30, 0.60]]))
        ht = ht.replace('flu-randomtokenrep,', ''.join(['flu-randomtokenrep-A[seed],'.replace('A', str(p)) for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]))
        ht = ht.replace('flu-sentencemiddleswap,', ''.join(['flu-sentencemiddleswap-A[seed],'.replace('A', str(p)) for p in [1, 2, 3]]))  
        ht = ht.replace('flu-lemmatizeverb,', ''.join(['flu-lemmatizeverb-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   
        ht = ht.replace('flu-removepreposition,', ''.join(['flu-removepreposition-A[seed],'.replace('A', str(p)) for p in [0.4, 0.7, 1.0]]))   
        ht = ht.replace('flu-noisepunct,', ''.join(['flu-noisepunct-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   
        ht = ht.replace('flu-removestopwords,', ''.join(['flu-removestopwords-A[seed],'.replace('A', str(p)) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))   
        ht = ht.replace('flu-removearticle,', ''.join(['flu-removearticle-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))   

        if ht == 'con-all,':
            ht = 'con-switchsent,con-switchnoun,con-genericner,con-switchverb,con-replacesent,con-switchner,con-negate,con-bertdiverge,'
            logger.info('con-all detected, setting hypo_transform to %s', ht)
        ht = ht.replace('con-switchsent,', ''.join(['con-switchsent-A[seed],'.replace('A', str(p)) for p in [1, 2]]))
        ht = ht.replace('con-switchnoun,', ''.join(['con-switchnoun-A[seed],'.replace('A', str(p)) for p in [2, 7]]))
        ht = ht.replace('con-genericner,', ''.join(['con-genericner-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))
        ht = ht.replace('con-switchverb,', ''.join(['con-switchverb-A[seed],'.replace('A', str(p)) for p in [8]]))
        ht = ht.replace('con-replacesent,', ''.join(['con-replacesent-A[seed],'.replace('A', str(p)) for p in [1, 2]]))
        ht = ht.replace('con-switchner,', ''.join(['con-switchner-A[seed],'.replace('A', str(p)) for p in [1, 3]]))
        ht = ht.replace('con-negate,', ''.join(['con-negate-A[seed],'.replace('A', str(p)) for p in [0.5, 1.0]]))
        ht = ht.replace('con-bertdiverge,', ''.join(['con-bertdiverge-A[seed],'.replace('A', str(p)) for p in [0.1, 0.2, 0.3, 0.4]]))

        assert(ht.endswith(',')); ht = ht[:-1]
        logger.info('expanded hypo_transfrom: %s', ht)
        args.hypo_transform = ht

    scorer = Scorer(args, args.file, args.device, args.multi_ref)

    #if args.select_multi: # this was originally used to cherry pick a reference to get better scores than models, but now i decided to use randomworddrop as a weak baseline
    #    scorer.select_multi(args.select_multi_metric)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.mover_score:
        METRICS.append('mover_score')
    if args.rouge:
        METRICS.append('rouge')
    if args.grouge:
        METRICS.append('grouge')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prism:
        METRICS.append('prism')
    if args.comet: METRICS.append('comet')
    if args.cometqe: METRICS.append('cometqe')
    if args.unieval: METRICS.append('unieval')
    if args.prompt is not None:
        prompt = args.prompt
        assert prompt in ['bart_src', 'bart_ref', 'bart_cnn_src',
                          'bart_cnn_ref', 'bart_para_src', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    res_d = scorer.score(METRICS)

    res_d = reduce_mean(args, res_d, scorer.transform_info)

    for me in res_d:
        logger.info('=== BEGIN OF REPORT for %s ===', me)
        refa_score = res_d[me]['ref']['mean']
        for sn in res_d[me]:
            s_now = res_d[me][sn]['mean']
            logger.info('%s: %f ref-percentage: %f noise-ratio: %f std: %f', sn, s_now, (s_now - refa_score) / abs(refa_score) * 100.0, res_d[me][sn]['edit_ratio'], res_d[me][sn]['std'])

        logger.info('=== END OF REPORT for %s ===', me)

    if args.save_score:
        score_utils.save_res(res_d, '../score_saves/{}/'.format(args.task_name))

    if args.output is not None:
        scorer.save_data(args.output)

if __name__ == '__main__':
    main()
