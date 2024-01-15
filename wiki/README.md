## Preparation 

First let's create a virtual environment (here I use the name blindspots_wiki, but you can use any name you prefer). This environment is for open-ended tasks.
```
conda create -n blindspots_wiki python=3.8
conda activate blindspots_wiki
```
Install pytorch with GPU (use your system's CUDA version)
```
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install other packages via pip
```
pip install -r requirements.txt 
```

For using Spacy in the consistency checks:
```
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
pip install pyinflect
```

Download NLTK data:
```
python -m nltk.downloader popular
```

Please install a custom version of MAUVE with gpt2/bert restriction removed (see https://github.com/cloudygoose/blindspot_nlg/issues/2 for details):

```
git clone git@github.com:jackjyzhang/mauve.git
cd mauve
pip install -e .
```

## Data
To get full wikitext-103 data, run `python get_wiki_data.py`, which simply download from huggingface datasets and save it locally.

`data/refs1.txt` and `data/refs2.txt` are two non-overlapping sets of wikitext-103 paragraphs of size 1000 each. The text is already preprocessed for the ease of using. We will use `data/refs1.txt` as the gold hypothesis, and `data/refs1.txt` as the starting point to create the noised hypothesis.

## Fluency & Consistency Tests: 

Fluency & consistency tests are designed to be runned in a full pipeline that first generate noised hypotheses, and then eval metric scores on these hypotheses. The full details of running the fluency and consistency is available in `pipeline.py`, but the helper script `pipeline.sh` could make things easier:

To run all fluency and consistency tests, simply do
```sh
for metric in gpt-ppl mlm-ppl mauve-gpt2 mauve-roberta mauve-electra;
do
    ./pipeline.sh $metric ref con-all flu-all
done
```

The above bash scripts go through each metric in a for loop, and for each metric, it runs on the gold hypothesis (denoted `ref`), all consistency tests (denoted `con-all`), and all fluency tests (denoted `flu-all`).

The score will be saved in `score_saves/wiki/${metric_name}/${test_name}.json` by default.

To run individual tests, please refere to [README_tests.md](./README_tests.md) for details. For example, to run the Truncation tests with 10% of the data truncated, use the test name `flu-truncate-0.10`. Note that the hyperparameter determining the amount of truncation is variable, for example, `flu-truncate-0.20` is truncating 20%.

An example that runs the truncation 20% test on MAUVE-RoBERTa is:
```sh
PYTHONPATH=$PWD ./pipeline.sh mauve-roberta flu-truncate-0.20
```

### Plotting Fluency & Consistency Results
Use `plot.py` to plot fluency & consistency results. The important arguments are `--metric_results_dir` which is the dir that contains results `.json` files, and `--output_path` which specifies the output png path. For example:
```sh
mkdir plots
python plot.py --metric_results_dir score_saves/wiki/mauve-roberta --output_path plots/mauve-roberta.png --error_bar
```
plots test results for MAUVE-RoBERTa with error bars.

More information on plotting can be found in `plot.py`. There are more arguments that makes plots better looking, e.g. `--max_edr` to make plotting focused on 0 to a max noise-ratio and makes things bigger.

## All Other Tests

### Step 1: generate noised hypotheses

#### **The Positioned Error Test (aka the Distant Error Test) and the Repetition Test**

Use `repl_gen.py` to generate noised hypotheses:
```sh
python repl_gen.py --generation data/refs2.txt --op $OP
```

where `$OP` is one of ['prefix', 'middle', 'suffix', 'prefix-shuffle', 'middle-shuffle', 'suffix-shuffle', 'replast'] that determines the operation. The first 6 corresponds to the 6 error types are used for the positioned error test. `repleast` is used for the repetition test.

The naming convention here is a bit different to the paper. Here is a conversion table, where noise type is the name shown on the paper:

| Noise Type     | Operation      |
|----------------|----------------|
| Random-Start   | prefix         |
| Random-Middle  | middle         |
| Random-End     | suffix         |

By default, the noised hypothesis will be generated at `data/refs2_repl-{OP}-10.txt` where `{OP}` is the chosen operation.

#### **Attention Pattern Analysis for the Positioned Error Test**

Use two examples `data/refs2-1.1.txt` and `data/refs2-1.2.txt` on wiki103 to create attention distribution plot with `analysis_position.py`:
```sh
python analysis_position.py --models gpt2-large roberta-large --input_text data/refs2-1.1.txt
```
By default, the generated plots are saved in `analysis_plots/` directory.

#### **The Frequent N-Gram Test**

First, use `freq_get_dict.py` to get n-gram statistics on the wikitext-103 dataset. For example, getting the frequencies of 4grams:
```sh
python freq_get_dict.py --dataset data/wikitext-103-raw --output_path metadata/freqdict_wiki103train_4gram.pkl --gram 4
```
which will save results to a pickle object in `metadata/freqdict_wiki103train_4gram.pkl`.

Then, using this statistics to generate gibberish that consists of top-k ngrams with `freq_top_ngrams.py`. For example, the below code generates top-50 4-gram sequence:
```sh
python freq_top_ngrams.py --counter_path metadata/freqdict_wiki103train_4gram.pkl --gram 4 --topk 50
```
Note that the path in --counter_path is the pickle we produced in the last step. By default, the noised hypothesis is saved in `data/top_ngram/n1000_max256/{n}gram_topk{k}.txt` where `{n}` and `{k}` are specified by --gram and --topk, resp.

### Step 2: run metrics on noised hypotheses
#### **PPL variants**
Similar to MAUVE, PPL variants have a script to evaluate the scores, `eval_clm.py`. You can use it as follows:

GPT-PPL:
```sh
python eval_clm.py --model gpt2-large --generation $GEN --output_suffix $SUFFIX
```
MLM-PPL:
```sh
python eval_clm.py --model roberta-large -mlm --generation $GEN --output_suffix $SUFFIX
```
where `GEN` is the hypothesis to test, and `SUFFIX` is for deduplication of output file.

Note that by default, the batch size for MLM-PPL is 32 for efficiency. If this does not fit in your GPU, you can adjust the batch size for MLM-PPl using the `--mlm_batch_size` flag. More details are available in `eval_mauve.py`.

#### **MAUVE variants**
The code that runs MAUVE variants on a reference and hypothesis pair is in `eval_mauve.py`. However, it is more convenient to use the helper script `eval_mauve.sh`:
```sh
./eval_mauve.sh $FEAT_MODEL $REF $SUFFIX $GENS
```
Here are what the argument means:
- FEAT_MODEL: Options are ['gpt2', 'rbt', 'electra-l-disc']. The featurize model to use, which is gpt2-large, roberta-large, or electra-large.
- REF: reference text
- SUFFIX: name suffix of output file. This is to make sure there's no naming conflict in the result file.
- GENS: a list of hypotheses to feed in. They will be evaluated using the same reference specified by REF.
For example:
```sh
./eval_mauve.sh rbt data/refs1.txt _gold_hypo data/refs2.txt
```
produce the MAUVE-RoBERTa score of using `data/refs1.txt` as the reference, and `data/refs2.txt` as the hypothesis. It will output a file `mauve_gold_hypo.csv` in the same directory of the last hypothesis file. More details are available in `eval_mauve.py`.

In our experiments, MAUVE-ELECTRA (i.e., using `electra-l-disc` as the FEAT_MODEL) is the best-performing variant, while rbt and gpt2 has blind spots

#### **Example**
Below we provide a concrete example of how to reproduce the positioned error test for MAUVE-RoBERTa on the RANDOM-START noise type, which corresponds to the `prefix` operation
```sh
PYTHONPATH=$PWD python repl_gen.py --generation data/refs2.txt --op prefix

PYTHONPATH=$PWD ./eval_mauve.sh rbt data/refs1.txt _prefix data/refs2_repl-prefix-10.txt
```
This will produce `data/mauve_rbt-l_prefix.csv` which stores the MAUVE score and `data/div_rbt-l_prefix.png` which stores the divergence plot.

## Notes

All code shown in this README assumes the root directory for projet is the `wiki/` dir. You may need to prepend `PYTHONPATH=/path/to/wiki/` if not running code from the `wiki/` dir.

IMPORTANT: about StopIteration runtime error: https://github.com/RaRe-Technologies/gensim/issues/2438#issuecomment-644753776 .
Basically, you need to comment out ...(conda dir).../site-packages/pattern/text/\_\_init\_\_.py line 609. 
To get the conda lib path you can: 
```
from distutils.sysconfig import get_python_lib; print(get_python_lib())
```
