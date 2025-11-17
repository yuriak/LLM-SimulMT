#!/bin/bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
# Change following parameters to run different dataset, e.g. mustc_enzh, wmt15_deen, iwslt15_envi
SRC=de
TGT=en
dataset_name=wmt15_deen
RAW_DATA_PATH="${ROOT_DIR}/dataset/${dataset_name}/processed/"
OUTPUT_PATH="${ROOT_DIR}/dataset/${dataset_name}/trajectory/"
FASTALIGN_PATH="${ROOT_DIR}/tools/fast_align/"
mkdir -p ${OUTPUT_PATH}

echo "Generating alignment for ${SRC}-${TGT}"
python ${SRC_PATH}/data_curation/get_alignment.py \
    --file-type json \
    --split all \
    --input-path ${RAW_DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --lang ${SRC}-${TGT} \
    --fastalign-path ${FASTALIGN_PATH}

echo "Generating trajectory for train data"
python ${SRC_PATH}/data_curation/generate_trajectory.py \
    --src ${OUTPUT_PATH}/${SRC}-${TGT}.train.${SRC} \
    --tgt ${OUTPUT_PATH}/${SRC}-${TGT}.train.${TGT} \
    --alignment ${OUTPUT_PATH}/alignment.nontok.train.${SRC}-${TGT} \
    --forward ${OUTPUT_PATH}/forward.nontok.train.${SRC}-${TGT} \
    --reverse ${OUTPUT_PATH}/reverse.nontok.train.${SRC}-${TGT} \
    --output ${OUTPUT_PATH}/${SRC}-${TGT}.train.txt \
    --output-pickle ${OUTPUT_PATH}/train_all.pkl \
    --src-lang ${SRC} --tgt-lang ${TGT}

echo "Generating trajectory for valid data"
python ${SRC_PATH}/data_curation/generate_trajectory.py \
    --src ${OUTPUT_PATH}/${SRC}-${TGT}.valid.${SRC} \
    --tgt ${OUTPUT_PATH}/${SRC}-${TGT}.valid.${TGT} \
    --alignment ${OUTPUT_PATH}/alignment.nontok.valid.${SRC}-${TGT} \
    --output ${OUTPUT_PATH}/${SRC}-${TGT}.valid.txt \
    --output-pickle ${OUTPUT_PATH}/valid_all.pkl \
    --src-lang ${SRC} --tgt-lang ${TGT}
