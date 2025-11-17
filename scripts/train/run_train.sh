#!/bin/bash
export WANDB_PROJECT="ConvSimulMT"

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)
SCRIPT_PATH="${ROOT_PATH}/scripts"
SRC_PATH="${ROOT_PATH}/src"
DATA_PATH="${ROOT_PATH}/dataset/"
CKPT_PATH="${ROOT_PATH}/ckpt/"
ACCELERATE_CONFIG_FILE=~/.cache/huggingface/accelerate/a100_config.yaml

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
ckpt_path=${CKPT_PATH}/${ckpt_name}/
mkdir -p $ckpt_path

detokenizer_args=""
trajectory_args=""
bin_file=${SRC_PATH}/train/sft_offline_prompt.py
if [ "${prompt_type}" == "conv" ]; then
    bin_file=${SRC_PATH}/train/sft_conv_prompt.py
    trajectory_args="--use_sent_boundary --use_offset --use_merge"
    data_dir=${DATA_PATH}/${dataset_name}/trajectory/
    if [ "${tgt_lang}" == "Chinese" ]; then
        detokenizer_args="--src_detokenizer bpe --tgt_detokenizer bpe"
    fi
else
    data_dir=${DATA_PATH}/${dataset_name}/processed/
fi

epoch=1
PYTHONPATH=$SRC_PATH TOKENIZERS_PARALLELISM=true accelerate launch --config_file $ACCELERATE_CONFIG_FILE $bin_file \
 --data_path ${data_dir} \
 --model_name $model_name \
 --epochs $epoch \
 --quant \
 --fa2 \
 --gradient_checkpointing \
 --source_lang ${src_lang} \
 --target_lang ${tgt_lang} \
 --special_token_map ${sp_map} \
 --job_name ${ckpt_name} \
 --max_seq_length 1024 \
 --save_on steps \
 --save_steps 5000 \
 --output_path $ckpt_path --bsz 16 --grad_accum 2 ${detokenizer_args} ${trajectory_args}
