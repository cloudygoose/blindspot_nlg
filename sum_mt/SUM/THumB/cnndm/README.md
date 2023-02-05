# <img src="https://github.com/jungokasai/THumB/blob/master/figs/thumb.png" height="25" alt="thumb-up"> THumB for CNNDM Summarization
`cnndm_THumB-1.0.jsonl` provides THumB (**T**ransparent **Hum**an **B**enchmark) scores for summaries on the CNNDM dataset. The evaluation protocol is described in [Fabbri et al., 2021](https://arxiv.org/abs/2007.12626). `cnndm_references.jsonl` is human-generated reference summaries from [Kryściński et al., 2019](https://arxiv.org/abs/1908.08960).
| Category |  |
| ------ | ------ |
| coherence | **expert** evaluation average (1-5) |
| consistency | **expert** evaluation average (1-5) |
| fluency | **expert** evaluation average (1-5) |
| relevance | **expert** evaluation average (1-5) |
| human_score | average of all four scores above |
| expert_annotations | original evaluations of three experts |
| \*turker_annotations | original evaluations of four Turkers  |
| SYS | system (summarization model) | 
| seg_id | segmentation (example) ID | 
| set_id | integer ID in the whole test split (0-11489) | 
| hyp | hypothesis (predicted) summary from the model | 

\* Not available for Pegasus-dynamic-mix.

## Evaluated Models and Call for Better Model Development

We follow the spirit of [SacreBLEU](https://arxiv.org/abs/1804.08771) and applied automatic [detokenization](https://www.nltk.org/) and/or [truecasing](https://stanfordnlp.github.io/CoreNLP/truecase.html) to model outputs to get **clean, truecased, untokenized** text.

Future summarization model developers, **please please do provide clean, truecased, untokenized summaries**, independent of your preprocessing for reproducible (automatic) evaluations. I know ROUGE is case-insensitive, but the world is NOT!!

100 summaries are evaluated for every model.
| Model | Reference |
| ------ | ------ |
| Lead-3 | First three sentences |
| NEUSUM | [Zhou et al., 2018](https://arxiv.org/abs/1807.02305) |
| BanditSum | [Dong et al., 2018](https://arxiv.org/abs/1809.09672) |
| RNES | [Wu and Fu, 2018](https://arxiv.org/abs/1804.07036) |
| Pointer-Generator | [See et al., 2017](https://arxiv.org/abs/1704.04368) |
| Fast-abs-rl | [Chen and Bansal, 2018](https://arxiv.org/abs/1805.11080) | 
| Bottom-Up | [Gehrmann et al., 2018](https://arxiv.org/abs/1808.10792) |
| Improve-abs | [Kryściński et al., 2018](https://arxiv.org/abs/1808.07913) |
| Unified-ext-abs | [Hsu et al., 2018](https://arxiv.org/abs/1805.06266) |
| ROUGESal | [Pasunuru and Bansal, 2018)](https://arxiv.org/abs/1804.06451) |
| Multi-task-Ent-QG | [Guo et al., 2018](https://arxiv.org/abs/1805.11004) |
| Closed-book-decoder | [Jiang and Bansal, 2018) ](https://arxiv.org/abs/1809.04585) |
| T5 | [Raffel et al., 2020](https://arxiv.org/abs/1910.10683) |
| GPT-2-zero-shot | [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593) |
| BART | [Lewis et al., 2020](https://arxiv.org/abs/1910.13461) |
| Pegasus-dynamic-mix | [Zhang et al., 2020](https://arxiv.org/abs/1912.08777) |
| Pegasus-huge-news | [Zhang et al., 2020](https://arxiv.org/abs/1912.08777) |
| Human-H | News article highlights |

## Citations
```
@article{fabbri2021summeval,
    title   = {{SummEval}: Re-evaluating Summarization Evaluation},
    author  = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
    journal = {TACL},
    year    = {2021},
    url     = {https://arxiv.org/abs/2007.12626},
}
@misc{kasai2021thumb,
    title   = {Transparent Human Evaluation for Image Captioning},
    author  = {Jungo Kasai and Keisuke Sakaguchi and Lavinia Dunagan and Jacob Morrison and Ronan Le Bras and Yejin Choi and Noah A. Smith},
    year    = {2021},
    url     = {https://arxiv.org/abs/2111.08940}, 
}
```
