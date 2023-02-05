This is the Repo for the paper: [On the Blind Spots of Model-Based Evaluation Metrics for Text Generation](https://arxiv.org/abs/2212.10020)

We have code for three tasks, summarization, translation, and open-ended generation. This README is for summarization or translation. For open-ended generation, please refer to ...

The steps for summarization or translation are almost identical. Below, we will primarily explain the steps for summarization.

## Preparation 

First let's create a virtual environment (here I use the name ss, but you can use any name you prefer). This environment is shared for summarization and translation tasks.
```
conda create -n ss python=3.8
conda activate ss
pip install -r requirements.txt 
```

For using Spacy in the consistency checks:
```
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

Goto the directory sum_mt/SUM (for MT, goto sum_mt/MT), first let's prepare some data:
```
bash prepare_data.sh 
```

To get the BARTScore-para model, please follow https://github.com/neulab/BARTScore/tree/main/SUM .

The current environment supports bart_score_cnn/para, bert_score, mover_score, comet, cometqe, unieval.

For prism (or prismqe), please do pip install fairseq==0.9.0 .

Note: If you only need to use our code for noise transformations, instead of reproducing our results for various metrics, then pretty much everything you need should be in sum_mt/sanity_transform.py and the files and packages it depends on. 

<!--
Then, to save the spacy results, call this in both wmt and sum dir:
```
python score.py --save_spacy_trf
```
-->

## Running Tests

Now we can begin to run stress tests! In general, the command looks like this:
```
python score.py --rouge --hypo_transform flu-truncate-0.1,flu-truncate-0.2,flu-truncate-0.3, --debug_transform
```
By default, it runs on the CNNDM dataset. For MT, the default dataset is WMT. To run the TED-MT dataset, please add --file ted_zhen/combined_zh-en.save.

The --rouge option means reporting on the (variants of) the ROUGE metric, it can be changed to --bert_score/comet/bart_score_cnn, etc.

The --hypo_transform option specifys what noise (and the amount of noise) we are adding to the gold hypotheses (which is reference data). Each noise specification is comma-seperated. In the example command, it means, adding 10/20/30% percentage of truncation.

The optional --debug_transform is to show examples of transformed hypotheses (for debugging).

The final output for rougeL-f should look like this:
```
=== BEGIN OF REPORT for rougeL-f ===
ref: 0.286197 ref-percentage: 0.000000 noise-ratio: 0.000000 std: 0.000000
ref_flu-truncate-0.1: 0.289952 ref-percentage: 1.311899 noise-ratio: 0.113948 std: 0.000000
ref_flu-truncate-0.2: 0.295400 ref-percentage: 3.215607 noise-ratio: 0.215171 std: 0.000000
ref_flu-truncate-0.3: 0.298050 ref-percentage: 4.141645 noise-ratio: 0.312281 std: 0.000000
=== END OF REPORT for rougeL-f ===
```

ref-percentage computes the percentage of score w.r.t. the gold (ref) score. Here, we observe as the noise-ratio becomes larger, the score increases instead of decreases, which means ROUGE-L fails this test.

For noise type that involves randomness, you can append "[seed]" to the noise type string, and the program will automatically run the noise transformation under 5 different seeds, for example:

```
python score.py --rouge --hypo_transform flu-randomworddrop-0.10[seed],flu-randomworddrop-0.20[seed],
```

We explain how to run **the tests involved in the paper [here](./README_tests.md)**.

For the exact implementation of each test, please refer to sum_mt/sanity_transform.py .

## Plotting

For the fluency/consistency tests, we can plot how the metirc scores decrease as the noise ratio increases. First, add the --save_score flag when running tests, for example (in sum_mt/SUM):
```
python score.py --rouge --hypo_transform flu-all, --save_score
python score.py --rouge --hypo_transform con-all, --save_score
```
Here, flu/con-all means we run all predefined fluency/consistency tests.

Then, goto sum_mt, and do the plotting:
```
python plot.py score_saves/sum/rougeL-f/ --error_bar --max_edr 0.6 --prefix con- --name_suffix _con
python plot.py score_saves/sum/rougeL-f/ --error_bar --max_edr 0.6 --prefix flu- --name_suffix _flu
```

The figures will be in score_saves/sum/rougeL-f/.

## Notes

IMPORTANT: about StopIteration runtime error: https://github.com/RaRe-Technologies/gensim/issues/2438#issuecomment-644753776 .
Basically, you need to comment out ...(conda dir).../site-packages/pattern/text/\_\_init\_\_.py line 609. 
To get the conda lib path you can: 
```
from distutils.sysconfig import get_python_lib; print(get_python_lib())
```

<!--
To cache transformed hypos for BertDiverge (we cache this also to avoid package conflict):
```
python score.py --bleu --file ted_zhen/combined_zh-en.save --hypo_transform con-bertdiverge, --cache_hypo_transform
```
-->
