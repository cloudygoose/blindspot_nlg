We explain how to run the tests involved in the paper here.

For the exact implementation of each test, please refer to sum_mt/sanity_transform.py .

For general instructions, please refer to this [README](./README.md).

For the CNNDM(SUM) goto the directory sum_mt/SUM. For MT, goto sum_mt/MT. 

For MT, the default dataset is WMT (De-En). To run the TED-MT dataset, please add --file ted_zhen/combined_zh-en.save. It's not hard to edit the code to run on other datasets.

## The Injection Test

The injection test is designed for UniEval, below is an example command:
```
python score.py --unieval --hypo_transform injection-unieval-ayes0,injection-unieval-amyes2, --debug_transform
```
amyes2 refers to the injection "Answer: Yes, this is a really good summary.", and ayes0 refers to the injection "Answer: Yes, this is a really coherent and consistent summary. And yes, it is relevant.". You can implement new injection in sanity_transform.py (search for "injection-unieval-").

## The Frequent N-gram or Repetition or Copy-Source Test
For the frequent n-gram test in the summarization test, we need to preprocess the data first, goto dir sum_mt/plm_ft, and do the following:
```
python get_data.py
python bart_ft.py --command freq
cd ../SUM
```

The following example command runs the frequent n-gram command:
```
python score.py --rouge --hypo_transform freq3gram-top5-len10,freq3gram-top5-len20,freq4gram-top5-len10,freq4gram-top5-len20 --debug_transform
```
top5-len10 means using the concatenation of 10 top-5 ngrams in the data.

For the repetition test, below is an example command (the last 4-gram is repeated for 5/10/20 times):
```
python score.py --bart_score_cnn --hypo_transform rep-span-5,rep-span-10,rep-span-20,  --debug_transform
```

For the copy-source test, below is an example command:
```
python score.py --bart_score_cnn --hypo_transform copysrc,  --debug_transform
```

## Fluency and Consistency Tests

There are several ways to run fluency/consistency tests:

You can specify a list of hyper-parameters, below, we use the truncation test as an example:
```
python score.py --rouge --hypo_transform flu-truncate-0.10,flu-truncate-0.20,flu-truncate-0.30,
```
Like we suggest in the paper, the gap of noise-ratio between two noise level should not be too small (we recommend close to or larger than 5%).

You can also simply specify:
```
--hypo_transform flu-truncate,
```
In this case, it will be automatically expanded to a predefined list of hyperparameters, in this case 0.1/0.2/0.3/0.4/0.5. You can of course, change the predefined list in score.py .

Finally, you can use the short hand flu-all, or con-all, to run a list of predefined tests. For example, the following command will use all of the implemented fluency tests for ROUGE.
```
python score.py --rouge --hypo_transform flu-all, --save_score
```

For how to plot the results (saved by the --save_score option), please see the plotting section in [here](./README.md).

The naming of the tests in the code is a bit different from the paper, we give the mapping list below.
 
Below are fluency tests:

|Name-Paper|Name-Code(Example)|Explanation|
|--|--|--|
|Truncation|flu-truncate-0.10|A portion (10%) of tokens at the end of the hypothesis are removed.|
|Article Removal|flu-removearticle-0.5|A random portion (50%) of articles (the/a/an) in the hypothesis are removed.|
|Preposition Removal|flu-removepreposition-0.5|A random portion (50%) of prepositions are removed.|
|Stop-word Removal|flu-removestopwords-0.2|A random portion (20%) of stop-words are removed.|
|Verb Lemmatization|flu-lemmatizeverb-0.5|A random portion (50%) of verbs in the hypothesis are lemmatized.|
|Token Drop|flu-randomworddrop-0.10|A random portion (10%) of tokens are removed.|
|Repeated Token|flu-randomtokenrep-0.10|A random portion (10%) of tokens are repeated once.|
|Local Swap|flu-randomlocalswap-0.10|A random portion (10%) of tokens are swapped with the token to the right of it.|
|Middle Swap|flu-sentencemiddleswap|The left and right part of the sentence is swapped (The cut-off point is right in the middle of the length). This is to synthesize a wrong subject-verb-object (SVO) order.|
|Noised Punctuation|flu-noisepunct-0.5|A random portion (50%) of the punctuations are noised. For example, commas are replaced by periods and vice versa.|

Below are consistency tests:

|Name-Paper|Name-Code(Example)|Explanation|
|--|--|--|
|Sentence Switching|con-switchsent-2|Several (2) random pairs of sentences in the hypothesis are switched, breaking temporal/logical order|
|Sentence Replacement|con-replacesent-1|Several (1) sentences in the hypothesis are replaced by a random irrelevant sentence.|
|Negation|con-negate-0.5|A random portion (50%) of sentences are negated.|
|Generic Named Entity|con-genericner-0.5|A random portion (50%) of the named entities in the hypothesis are replaced by a generic phrase, destroying the information.|
|Named Entity Switching|con-switchner-1|Several (1) random pairs of named entities in the hypothesis are switched, breaking factuality.|
|Verb Switching|con-switchverb-2|Several (2) random pairs of verbs in the hypothesis are switched.|
|Noun Switching|con-switchnoun-2|Several (2) random pairs of nouns in the hypothesis are switched.|
|BERT-diverge|con-bertdiverge-0.1|A random portion (10%) of the tokens in the hypothesis are replaced one by one by sampling from the top-10 prediction of a masked language model (RoBERTa). At each step, one token at a random position is replaced by [MASK], and inputed to RoBERTa for prediction. Since this process do not have access to the source text, the semantics of the hypothesis would gradually diverge.|

### Notes

IMPORTANT: about StopIteration runtime error: https://github.com/RaRe-Technologies/gensim/issues/2438#issuecomment-644753776 .
Basically, you need to comment out ...(conda dir).../site-packages/pattern/text/\_\_init\_\_.py line 609. 
To get the conda lib path you can: 
```
from distutils.sysconfig import get_python_lib; print(get_python_lib())
```


