'''copied from /data1/groups/txml/projects/metricnlg_2205/BARTScore/utils.py'''
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
nltk_detokenizer = Detok()
from nltk import word_tokenize
import re

def nltk_detokenize(text):
    tt = word_tokenize(text)
    text = nltk_detokenizer.detokenize(tt)
    text = re.sub('\s*,\s*', ', ', text)
    text = re.sub('\s*\.\s*', '. ', text)
    text = re.sub('\s*\?\s*', '? ', text)
    text = text.strip()
    return text