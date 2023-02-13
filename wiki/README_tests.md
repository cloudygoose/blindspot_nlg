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