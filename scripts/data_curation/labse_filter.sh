#!/bin/bash
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
# Change following parameters to run different dataset
SRC=de
TGT=en
RAW_DATA_PATH="${ROOT_DIR}/dataset/wmt15_deen/raw/"
OUTPUT_PATH="${ROOT_DIR}/dataset/wmt15_deen/processed/"
mkdir -p ${OUTPUT_PATH}

GPU_COUNT=4
for i in {0..${GPU_COUNT-1}}; do
    nohup bash -c "python ${SRC_PATH}/data_curation/labse_filter.py --src ${RAW_DATA_PATH}/train.${SRC} --tgt ${RAW_DATA_PATH}/train.${TGT} --output-path ${OUTPUT_PATH} --cuda ${i}:${GPU_COUNT}" > ${i}.log &
done

echo "LabSE filter completed"

# Merge the filtered data
# for i in {0..${GPU_COUNT-1}}; do
#     cat ${OUTPUT_PATH}/${i}.src >> ${OUTPUT_PATH}/train.${SRC}
#     cat ${OUTPUT_PATH}/${i}.tgt >> ${OUTPUT_PATH}/train.${TGT}
# done

# Convert to json
# bash $SCRIPT_PATH/data_curation/txt_to_json.sh ${OUTPUT_PATH}/train.${SRC} ${OUTPUT_PATH}/train.${TGT} ${OUTPUT_PATH}/train.json
# bash $SCRIPT_PATH/data_curation/txt_to_json.sh ${RAW_DATA_PATH}/valid.${SRC} ${RAW_DATA_PATH}/valid.${TGT} ${OUTPUT_PATH}/valid.json