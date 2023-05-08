![](teaser.png)

This is the Repo for the paper: [On the Blind Spots of Model-Based Evaluation Metrics for Text Generation](https://arxiv.org/abs/2212.10020)

We have code for three tasks, summarization, translation, and open-ended generation. For summarization or translation, please see [here](sum_mt/README.md). 

Our code and data for open-ended generation is [here](wiki/README.md). Note that the package dependency for open-ended generation is different (The metrics are different and the code is not shared), so you might need to create a seperate virtual environment.

One shortcoming of WMT data is they mostly contain only one sentence, therefore we build the TED-MT dataset. For information about how the **TED-MT** data is constructed, please refer to sum_mt/MT/ted_zhen/README.txt . 

The code for summarization or translation is developed based on the released code of [BARTScore](https://github.com/neulab/BARTScore/).

To appear at ACL 2023. If you find our work useful, please cite our paper, thanks!
```
@misc{he2023blind,
      title={On the Blind Spots of Model-Based Evaluation Metrics for Text Generation}, 
      author={Tianxing He and Jingyu Zhang and Tianle Wang and Sachin Kumar and Kyunghyun Cho and James Glass and Yulia Tsvetkov},
      year={2023},
      eprint={2212.10020},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
