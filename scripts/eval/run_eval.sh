#!/bin/bash

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)
SCRIPT_PATH="${ROOT_PATH}/scripts"
SRC_PATH="${ROOT_PATH}/src"
DATA_PATH="${ROOT_PATH}/dataset"
CKPT_PATH="${ROOT_PATH}/ckpt"


# Change following parameters to run different prompt type, dataset, model, etc.


# Language pairs
src=en
# src=de
# src=en
tgt=zh
# tgt=en
# tgt=vi
lang_pair=${src}-${tgt}

src_lang=English
# src_lang=German
# tgt_lang=English
tgt_lang=Chinese
# tgt_lang=English
# tgt_lang=Vietnamese

# Dataset name
dataset_name=mustc_enzh
# dataset_name=wmt15_deen
# dataset_name=iwslt15_envi
data_dir=${DATA_PATH}/${dataset_name}/

# Prompt type
prompt_type="conv"
# prompt_type="offline"

# Models
model_name="meta-llama/Llama-2-7b-chat-hf"
# model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_name="meta-llama/Llama-3.2-1B-Instruct"
# model_name="Qwen/Qwen2.5-7B-Instruct"
model_short_name=l27
# model_short_name=l318
# model_short_name=l321
# model_short_name=q257
sp_map="llama2"
# sp_map="llama3"
# sp_map="qwen"

ckpt_name="${model_short_name}_${dataset_name}_${prompt_type}_prompt"
ckpt=${CKPT_PATH}/${ckpt_name}/

submit_job_script=${SCRIPT_PATH}/submit_infer_job.sh
common_sbatch_script=${SCRIPT_PATH}/eval/${prompt_type}/eval_sbatch.sh

# With RALCP (beam size must be > 1, acr is used)
n=(3 5 7 9 11 13 15 999)
beam=5
acr=0.5
for n in ${n[@]}; do
    sbatch_name=${model_short_name}-${lang_pair}-${n}-${beam}-${acr}
    sbatch -J ${sbatch_name} ${submit_job_script} ${common_sbatch_script} \
        -P ${prompt_type} \
        -d ${dataset_name} \
        -j ${ckpt_name} \
        -n ${n} \
        -b ${beam} \
        -a ${acr} \
        -D ${data_dir} \
        -l ${lang_pair} \
        -s ${src_lang} \
        -t ${tgt_lang} \
        -m ${model_name} \
        -c ${ckpt} \
        -p ${sp_map}
done

# Without RALCP (beam size must be 1, acr is not used actually)
n=(3 5 7 9 11 13 15 999)
beam=1
acr=0.5

for n in ${n[@]}; do
    sbatch_name=${model_short_name}-${lang_pair}-${n}-${beam}-${acr}
    sbatch -J ${sbatch_name} ${submit_job_script} ${common_sbatch_script} \
        -P ${prompt_type} \
        -d ${dataset_name} \
        -j ${ckpt_name} \
        -n ${n} \
        -b ${beam} \
        -a ${acr} \
        -D ${data_dir} \
        -l ${lang_pair} \
        -s ${src_lang} \
        -t ${tgt_lang} \
        -m ${model_name} \
        -c ${ckpt} \
        -p ${sp_map}
done