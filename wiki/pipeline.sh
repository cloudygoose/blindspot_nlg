#!/bin/bash
module load anaconda/2021a; source activate jack

DATA_DIR=data/wikitext-103-raw/train
OUTPUT_DIR=score_saves/wiki
REF_NUM=1000 # 1000 default
MAX_LEN=256

if [[ $REF_NUM -ne 1000 ]]; then
    OUTPUT_DIR+="_ref$REF_NUM"
fi
METRICS=($1) # (gpt-ppl mlm-ppl mauve-gpt2 mauve-roberta mauve-electra)
OP_NAMES=(${@:2}) # (ref con-all flu-all con-negate-A con-switchsent-A con-replacesent-A con-genericner)

echo "output_dir: $OUTPUT_DIR"
echo "metrics: ${METRICS[@]}"
echo "op_names: ${OP_NAMES[@]}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
time python pipeline.py \
    $DATA_DIR $OUTPUT_DIR --ref_num $REF_NUM --maxlen $MAX_LEN \
    --metrics ${METRICS[@]} \
    --op_names ${OP_NAMES[@]} \
    --seeds 0,1,2,3,4 \