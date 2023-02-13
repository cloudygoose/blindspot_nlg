#!/bin/bash

FEAT_MODEL=$1 # gpt2, rbt, electra-l-disc
REF=$2
SUFFIX=$3
GENS=(${@:4}) 

if [[ $FEAT_MODEL == 'gpt2' ]]; then
    params="--output_suffix gpt2-l$SUFFIX"
elif [[ $FEAT_MODEL == 'rbt' ]]; then
    params="--output_suffix rbt-l$SUFFIX --feature_extractor roberta-large"
elif [[ $FEAT_MODEL == 'electra-l-disc' ]]; then
    params="--output_suffix electra-l-disc$SUFFIX --feature_extractor google/electra-large-discriminator"
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python eval_mauve.py -r $REF -g ${GENS[@]} $params
# python eval_mauve.py -r gen/n1000_max256/reference_train/reference.txt -fc -g $NAME $params
