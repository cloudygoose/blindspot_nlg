#source activate metric 

echo caching spacy results
mkdir -p spacy_saves
python score.py --save_spacy_trf #for WMT
python score.py --save_spacy_trf --file ted_zhen/combined_zh-en.save
sleep 5

echo download comet
cd models
python download_comet.py
cd ..
sleep 5

echo download prism
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
mv m39v1 models/
rm m39v1.tar

echo cache transformed hypos for BertDiverge
mkdir -p hypotransform_saves
python score.py --bleu --hypo_transform con-bertdiverge, --cache_hypo_transform
python score.py --bleu --file ted_zhen/combined_zh-en.save --hypo_transform con-bertdiverge, --cache_hypo_transform
sleep 5
