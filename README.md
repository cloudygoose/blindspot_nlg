This is the Repo for the paper: [On the Blind Spots of Model-Based Evaluation Metrics for Text Generation](https://arxiv.org/abs/2212.10020)

We have code for three tasks, summarization, translation, and open-ended generation. For summarization or translation, please see [here](sum_mt/README.md). 

Our code and data for open-ended generation will be out soon. Note that the package dependency for open-ended generation is different (The metrics are different and the code is not shared), so you might need to create a seperate virtual environment.

One shortcoming of WMT data is they mostly contain only one sentence, therefore we build the TED-MT dataset. For information about how the **TED-MT** data is constructed, please refer to sum_mt/MT/ted_zhen/README.txt . 

The code for summarization or translation is developed based on the released code of [BARTScore](https://github.com/neulab/BARTScore/).

If you find our work useful, please cite our paper, thanks!
```
@misc{https://doi.org/10.48550/arxiv.2212.10020,
  doi = {10.48550/ARXIV.2212.10020},
  url = {https://arxiv.org/abs/2212.10020},
  author = {He, Tianxing and Zhang, Jingyu and Wang, Tianle and Kumar, Sachin and Cho, Kyunghyun and Glass, James and Tsvetkov, Yulia},
  title = {On the Blind Spots of Model-Based Evaluation Metrics for Text Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
