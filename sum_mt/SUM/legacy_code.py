### from score.py, originally for select a ref for the original reference, however, then i found that the summeval should use multiref for eval
### so i changed it and moved to select_multi.py
    
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
        ref_lines = self.single_ref_lines if not self.multi_ref else self.multi_ref_lines
        ref_lines = [detokenize(line) for line in ref_lines]

        multi_refs = self.multi_ref_lines_reserve
        src_lines = self.get_src_lines() #source documents
        src_lines = [detokenize(line) for line in src_lines]
        
        best_scores, best_idx, all_scores = [-1000 for kk in ref_lines], [0 for kk in ref_lines], [[] for kk in ref_lines]
        for idx in range(len(multi_refs[0])):
            sys_lines = [rr[idx] for rr in multi_refs]
            sys_lines = [detokenize(line) for line in sys_lines]
            
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
                    s = scorer.score(ref_lines[i], sys_lines[i])
                    s_now.append(s[rouge_var].fmeasure)

            if metric_name.startswith('bart_score'):
                src_hypo = bart_scorer.score(src_lines, sys_lines, batch_size=4)
                s_now = src_hypo

            for i in range(len(ref_lines)):
                all_scores[i].append(s_now[i])
                if s_now[i] > best_scores[i]:
                    best_scores[i] = s_now[i]
                    best_idx[i] = idx
        
        best_refs = [multi_refs[i][best_idx[i]] for i in range(len(ref_lines))]
        logger.info('score now: %f', np.mean(best_scores))
        save_fn = f'select_multiref/best_ref_{metric_name}.save'
        logger.info('saving best_refs to %s, will wait 5 seconds', save_fn); time.sleep(6);
        torch.save(best_refs, save_fn)
        breakpoint()
        x=1+1