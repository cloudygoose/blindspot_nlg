import argparse
import os
import time
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
import sanity_transform

REF_HYPO = read_file_to_list('files/tiny_ref_hypo_prompt.txt')

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(); logger.handlers = [];
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

class Scorer:
    """ Support BLEU, CHRF, BLEURT, PRISM, COMET, BERTScore, BARTScore """

    def __init__(self, args, file_path, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.args = args
        self.device = device

        logger.info(f'Data loaded from {file_path}.')
        if file_path.endswith('.pkl'):
            self.data = read_pickle(file_path)
        if file_path.endswith('.save'):
            self.data = torch.load(file_path)

        self.refs, self.betters, self.worses, self.srcs = [], [], [], []
        for doc_id in self.data:
            self.refs.append(self.data[doc_id]['ref'])
            self.srcs.append(self.data[doc_id]['src'])
            self.betters.append(self.data[doc_id]['better']['sys'])
            if 'worse' in self.data[doc_id]:
                self.worses.append(self.data[doc_id]['worse']['sys'])

    def save_data(self, path):
        save_pickle(self.data, path)

    def record(self, scores_better, scores_worse, name):
        """ Record the scores from a metric """
        for doc_id in self.data:
            self.data[doc_id]['better']['scores'][name] = str(scores_better[doc_id])
            self.data[doc_id]['worse']['scores'][name] = str(scores_worse[doc_id])

    def score(self, metrics):
        for metric_name in metrics:
            if metric_name == 'bleu':
                from sacrebleu import corpus_bleu
                from sacremoses import MosesTokenizer

                def run_sentence_bleu(candidates: list, references: list) -> list:
                    """ Runs sentence BLEU from Sacrebleu. """
                    tokenizer = MosesTokenizer(lang='en')
                    candidates = [tokenizer.tokenize(mt, return_str=True) for mt in candidates]
                    references = [tokenizer.tokenize(ref, return_str=True) for ref in references]
                    assert len(candidates) == len(references)
                    bleu_scores = []
                    for i in range(len(candidates)):
                        bleu_scores.append(corpus_bleu([candidates[i], ], [[references[i], ]]).score)
                    return bleu_scores

                start = time.time()
                print(f'Begin calculating BLEU.')
                scores_better = run_sentence_bleu(self.betters, self.refs)
                scores_worse = run_sentence_bleu(self.worses, self.refs)
                print(f'Finished calculating BLEU, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bleu')

            elif metric_name == 'chrf':
                from sacrebleu import sentence_chrf

                def run_sentence_chrf(candidates: list, references: list) -> list:
                    """ Runs sentence chrF from Sacrebleu. """
                    assert len(candidates) == len(references)
                    chrf_scores = []
                    for i in range(len(candidates)):
                        chrf_scores.append(
                            sentence_chrf(hypothesis=candidates[i], references=[references[i]]).score
                        )
                    return chrf_scores

                start = time.time()
                print(f'Begin calculating CHRF.')
                scores_better = run_sentence_chrf(self.betters, self.refs)
                scores_worse = run_sentence_chrf(self.worses, self.refs)
                print(f'Finished calculating CHRF, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'chrf')

            elif metric_name == 'bleurt':
                from bleurt import score

                def run_bleurt(
                        candidates: list, references: list, checkpoint: str = "models/BLEURT-20" #"models/bleurt-large-512"
                ):
                    scorer = score.BleurtScorer(checkpoint)
                    scores = scorer.score(references=references, candidates=candidates)
                    return scores

                start = time.time()
                print(f'Begin calculating BLEURT.')
                scores_better = run_bleurt(self.betters[:200], self.refs[:200])
                scores_worse = run_bleurt(self.worses[:200], self.refs[:200])
                print(f'Finished calculating BLEURT, time passed {time.time() - start}s.')
                breakpoint()
                self.record(scores_better, scores_worse, 'bleurt')

            elif metric_name == 'prism':
                from prism import Prism

                def run_prism(mt: list, ref: list) -> list:
                    prism = Prism(model_dir="./models/m39v1", lang='en', temperature=1.0)
                    _, _, scores = prism.score(cand=mt, ref=ref, segment_scores=True)
                    return list(scores)

                def run_prism_qe(mt: list, src: list) -> list:
                    prism = Prism(model_dir="./models/m39v1", lang='en', temperature=1.0)
                    scores = prism.score(cand=mt, src=src, segment_scores=True)
                    return list(scores)

                start = time.time()
                print(f'Begin calculating PRISM.')
                scores_better = run_prism(self.betters, self.refs)
                scores_worse = run_prism(self.worses, self.refs)
                #scores_better = run_prism_qe(self.betters[:200], self.srcs[:200])
                #scores_worse = run_prism_qe(self.worses[:200], self.srcs[:200])
                print(f'Finished calculating PRISM, time passed {time.time() - start}s.')
                logger.info('better: %f worse: %f', np.mean(scores_better), np.mean(scores_worse))
                #self.record(scores_better, scores_worse, 'prism')

            elif metric_name == 'comet' or metric_name == 'cometqe':
                from comet import load_from_checkpoint
                args = self.args
                def create_samples():
                    """ Dataframe to dictionary. """
                    hyp1_samples, hyp2_samples = [], []
                    for doc_id in self.data:
                        hyp1_samples.append(
                            {
                                'src': str(self.data[doc_id]['src']),
                                'ref': str(self.data[doc_id]['ref']),
                                'mt': str(self.data[doc_id]['better']['sys'])
                            }
                        )
                        #hyp2_samples.append(
                        #    {
                        #        'src': str(self.data[doc_id]['src']),
                        #        'ref': str(self.data[doc_id]['ref']),
                        #        'mt': str(self.data[doc_id]['worse']['sys'])
                        #    }
                        #)
                    return hyp1_samples, hyp2_samples 

                #checkpoint = './models/wmt-large-da-estimator-1718/_ckpt_epoch_1.ckpt'
                #model_path = '/home/gridsan/tianxing/txml_shared/projects/metricnlg_2205/BARTScore/WMT/models/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt'
                #model = load_from_checkpoint(model_path)
                if metric_name == 'comet':
                    model_path = 'models/wmt20-comet-da.save'
                if metric_name == 'cometqe':
                    model_path = 'models/wmt21-comet-qe-mqm.save' #for this the 'ref' won't be used
                print('loading from', model_path)
                model = torch.load(model_path)
                hyp1_samples, hyp2_samples = create_samples()
                refs = [ss['ref'] for ss in hyp1_samples]
                transform_d = {'refs': refs}      

                logger.info('debug: only do 2000 samples for speed!')
                hyp1_samples, hyp2_samples = hyp1_samples[:2000], hyp2_samples[:2000]

                breakpoint()
                if args.hypo_transform is not None:
                    for kk, sample in enumerate(hyp1_samples):
                        sample['mt'] = sanity_transform.sanity_transform(sample['mt'], args.hypo_transform, src_line = sample['src'], idx = kk, transform_d = transform_d)

                start = time.time()
                print(f'Begin calculating COMET.')
                scores_better_seg, scores_better = model.predict(hyp1_samples)

                #scores_worse_seg, scores_worse = model.predict(hyp2_samples)
                print(f'Finished calculating COMET, time passed {time.time() - start}s.')
                logger.info('scores for hyp1: %f', scores_better)
                breakpoint()
                xx = 1+1
                #self.record(scores_better, scores_worse, 'comet')

            elif metric_name == 'bert_score':
                import bert_score

                def run_bertscore(mt: list, ref: list):
                    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
                    _, _, f1 = bert_score.score(
                        cands=mt,
                        refs=ref,
                        idf=False,
                        batch_size=32,
                        lang='en',
                        rescale_with_baseline=False,
                        verbose=True,
                        nthreads=4,
                    )
                    return f1.numpy()

                start = time.time()
                print(f'Begin calculating BERTScore.')
                scores_better = run_bertscore(self.betters, self.refs)
                scores_worse = run_bertscore(self.worses, self.refs)
                print(f'Finished calculating BERTScore, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bert_score')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                from bart_score import BARTScorer

                def run_bartscore(scorer, mt: list, ref: list):
                    hypo_ref = np.array(scorer.score(mt, ref, batch_size=4))
                    ref_hypo = np.array(scorer.score(ref, mt, batch_size=4))
                    avg_f = 0.5 * (ref_hypo + hypo_ref)
                    return avg_f

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                start = time.time()
                print(f'Begin calculating BARTScore.')
                scores_better = run_bartscore(bart_scorer, self.betters, self.refs)
                scores_worse = run_bartscore(bart_scorer, self.worses, self.refs)
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, metric_name)

            elif metric_name.startswith('prompt'):
                """ BARTScore adding prompts """
                from bart_score import BARTScorer

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

                start = time.time()
                print(f'BARTScore-P setup finished. Begin calculating BARTScore-P.')
                for prompt in tqdm(REF_HYPO, total=len(REF_HYPO), desc='Calculating prompt.'):
                    ref_better_en = np.array(bart_scorer.score(suffix_prompt(self.refs, prompt), self.betters,
                                                               batch_size=4))
                    better_ref_en = np.array(bart_scorer.score(suffix_prompt(self.betters, prompt), self.refs,
                                                               batch_size=4))

                    better_scores = 0.5 * (ref_better_en + better_ref_en)

                    ref_worse_en = np.array(bart_scorer.score(suffix_prompt(self.refs, prompt), self.worses,
                                                              batch_size=5))
                    worse_ref_en = np.array(bart_scorer.score(suffix_prompt(self.worses, prompt), self.refs,
                                                              batch_size=5))
                    worse_scores = 0.5 * (ref_worse_en + worse_ref_en)
                    self.record(better_scores, worse_scores, f'{name}_en_{prompt}')

                    ref_better_de = np.array(bart_scorer.score(self.refs, prefix_prompt(self.betters, prompt),
                                                               batch_size=5))
                    better_ref_de = np.array(bart_scorer.score(self.betters, prefix_prompt(self.refs, prompt),
                                                               batch_size=5))
                    better_scores = 0.5 * (ref_better_de + better_ref_de)

                    ref_worse_de = np.array(bart_scorer.score(self.refs, prefix_prompt(self.worses, prompt),
                                                              batch_size=5))
                    worse_ref_de = np.array(bart_scorer.score(self.worses, prefix_prompt(self.refs, prompt),
                                                              batch_size=5))
                    worse_scores = 0.5 * (ref_worse_de + worse_ref_de)
                    self.record(better_scores, worse_scores, f'{name}_de_{prompt}')
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bleu', action='store_true', default=False,
                        help='Whether to calculate BLEU')
    parser.add_argument('--chrf', action='store_true', default=False,
                        help='Whether to calculate CHRF')
    parser.add_argument('--bleurt', action='store_true', default=False,
                        help='Whether to calculate BLEURT')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--comet', action='store_true', default=False,
                        help='Whether to calculate COMET')
    parser.add_argument('--cometqe', action='store_true', default=False,
                        help='Whether to calculate COMET-QE')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Whether to calculate BARTScore-P, can be bart_ref, bart_cnn_ref, bart_para_ref')
    parser.add_argument('--hypo_transform', type=str, default=None, help='transform the hypo (system output) for sanity check purposes')

    args = parser.parse_args()

    scorer = Scorer(args, args.file, args.device)

    METRICS = []
    if args.bleu:
        METRICS.append('bleu')
    if args.chrf:
        METRICS.append('chrf')
    if args.bleurt:
        METRICS.append('bleurt')
    if args.prism:
        METRICS.append('prism')
    if args.comet:
        METRICS.append('comet')
    if args.cometqe:
        METRICS.append('cometqe')
    if args.bert_score:
        METRICS.append('bert_score')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prompt is not None:
        prompt = args.prompt
        assert prompt in ['bart_ref', 'bart_cnn_ref', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()

"""
python score.py --file kk-en/data.pkl --device cuda:0 --output kk-en/scores.pkl --bleu --chrf --bleurt --prism --comet --bert_score --bart_score --bart_score_cnn --bart_score_para

python score.py --file lt-en/scores.pkl --device cuda:3 --output lt-en/scores.pkl --bart_score --bart_score_cnn --bart_score_para
"""
