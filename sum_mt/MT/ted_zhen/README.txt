The raw data is from https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/ .
The select_id_test.txt is based on ted_test1_en-zh.raw
We found that in the raw data, the sentence splitting contains error in quotes. Therefore we avoid those in the selections.
The combine_data.py is used for combining the original English reference, with our translation. Note that we also correct some error in the original Chinese, so we will use the Chinese lines in translate/zh_en_jacktx.txt.
The combined data for the score.py to load is combined_zh-en.save

Tianxing & Jingyu & Tianle