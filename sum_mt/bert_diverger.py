import copy
import torch
import argparse
import random
import numpy as np
import editdistance

import logging
import bert_score

from transformers import (AutoModel, AutoTokenizer, BertModel, RobertaTokenizer, RobertaForMaskedLM)
from transformers import logging

punctuation = ['.', ':', ',', '/', '?', '<', '>', ';', '[', ']', '{', '}', '-', '_', '`', '~', '+', '=', '\'', '\"', '|', '\\']

def prepare_sentence(tokenizer, text, id):
    ids = []
    ids.append(tokenizer._convert_token_to_id(tokenizer.bos_token))
    for i in range(len(text)):
        if i == id:
            ids.append(tokenizer._convert_token_to_id(tokenizer.mask_token))
        else:
            for bpe_token in tokenizer.bpe(text[i]).split(" "):
                ids.append(tokenizer._convert_token_to_id(bpe_token))
    ids.append(tokenizer._convert_token_to_id(tokenizer.eos_token))
    return torch.tensor([ids]).long()

# The BertDiverge consistency check
class BertDiverger:
    def __init__(self, top_k = 10):
        self.top_k = top_k

        model_type = "roberta-large"  # default #./roberta-large for older version
        self.tokenizer = RobertaTokenizer.from_pretrained(model_type, use_fast=False, do_lower_case=True)
        self.model = RobertaForMaskedLM.from_pretrained(model_type)
        self.model.eval()
        self.model = self.model.cuda()

    def modify(self, line, ratio):
        import regex as re
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokenized_text = []
        for token in re.findall(self.pat, line):
            token = "".join(
                self.tokenizer.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            #tokenized_text.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
            tokenized_text.append(token)
        #tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        num = int(round(ratio * length))
        if num == 0:
            return line

        arr = np.array(list(range(length)))
        arr = np.random.permutation(arr)
        # sort and change
        cnt = 0
        while cnt < num:
            if cnt == length:
                break
            id = arr[cnt]
            cnt += 1
            bpe_check = len(self.tokenizer.bpe(tokenized_text[id]).split(" "))
            if bpe_check > 1:
                num += 1
                continue
            ids = prepare_sentence(self.tokenizer, tokenized_text, id)
            with torch.no_grad():
                output = self.model(ids.cuda())
            predictions = output[0]
            masked_index = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
            value, predicted_index = torch.topk(predictions[0, masked_index], k=self.top_k)
            value = value.cpu().numpy()
            value = np.exp(value)
            value = value / np.sum(value)
            predicted_index = predicted_index.cpu().numpy()
            select_index = np.random.choice(predicted_index, 1, p=value)
            predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in select_index]
            tokenized_text[id] = predicted_token[0]

        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

#bert_diverger = BertDiverger()
#print(bert_diverger.modify('I went to school to learn English.', 0.3))
#breakpoint()
