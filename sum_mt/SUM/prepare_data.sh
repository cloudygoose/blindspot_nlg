#source activate metric 

echo preprocess cnndm data, this step needs to be done first
cd files
python nltk_download.py
mkdir -p cnndm_saves
python get_data_cnndm.py
sleep 2
#du -sh cnndm_saves/test.json
echo the resulting test.json should be of size 49M
cd ..
sleep 5

echo caching spacy results
mkdir -p spacy_saves
python score.py --save_spacy_trf
sleep 5

echo download prism
bash download.sh
sleep 5

echo download comet
cd models
python download_comet.py
cd ..
sleep 5

echo prepare BertDiverge
mkdir -p hypotransform_saves
python score.py --rouge --hypo_transform con-bertdiverge, --cache_hypo_transform

