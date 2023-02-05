# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer

import logging
logger = logging.getLogger()

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, model_base = 'facebook/bart-large', checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length

        if model_base.startswith('facebook/bart'):
            mC, tC =  BartForConditionalGeneration, BartTokenizer; 
        if model_base.startswith('t5'):
            mC, tC =  T5ForConditionalGeneration, T5Tokenizer; 
        
        logger.info('BARTScorer ini loading tokenizer from %s', model_base)
        self.tokenizer = tC.from_pretrained(model_base) #, local_files_only = True)
        logger.info('BARTScorer ini loading checkpoint from %s', checkpoint)
        self.model = mC.from_pretrained(checkpoint) #, local_files_only = True)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self):
        """ Load model from paraphrase finetuning """
        logger.info('loading paraphrase finetuned model')
        self.model.load_state_dict(torch.load('models/bart.pth', map_location=self.device))
    
    def get_model_gens(self, srcs, batch_size):
        assert(len(srcs) % batch_size == 0)
        gen_lines = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            with torch.no_grad():
                encoded_src = self.tokenizer(
                    src_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ) #this is right padding
                src_tokens = encoded_src['input_ids'].to(self.device)
                src_mask = encoded_src['attention_mask'].to(self.device)
                summary_ids = self.model.generate(src_tokens, do_sample = False, num_beams=5, min_length=0, max_length = 100)
                for j in range(summary_ids.size(0)):
                    gen_lines.append(self.tokenizer.decode(summary_ids[j], skip_special_tokens=True))
        return gen_lines

    def score(self, srcs, tgts, batch_size):
        """ Score a batch of examples """
        assert(len(srcs) % batch_size == 0)
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    #loss = loss.sum(dim=1) #only for debug, a bartscore without average
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list
