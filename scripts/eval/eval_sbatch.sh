#!/bin/bash
ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)
SCRIPT_PATH="${ROOT_PATH}/scripts"
SRC_PATH="${ROOT_PATH}/src"

# Default values for arguments
prompt_type=""
dataset_name=""
job_name=""
n=""
beam=""
acr=""
data_dir=""
lang_pair=""
src_lang=""
tgt_lang=""
model_name=""
ckpt=""
sp_map=""

# Parse arguments
while getopts P:d:j:n:b:a:D:l:s:t:m:c:p: flag
do
    case "${flag}" in
        P) prompt_type=${OPTARG};;
        d) dataset_name=${OPTARG};;
        j) job_name=${OPTARG};;
        n) n=${OPTARG};;
        b) beam=${OPTARG};;
        a) acr=${OPTARG};;
        D) data_dir=${OPTARG};;
        l) lang_pair=${OPTARG};;
        s) src_lang=${OPTARG};;
        t) tgt_lang=${OPTARG};;
        m) model_name=${OPTARG};;
        c) ckpt=${OPTARG};;
        p) sp_map=${OPTARG};;
        *) echo "Invalid option"; exit 1;;
    esac
done

src_tokenizer="space"
if [ "$tgt_lang" == "Chinese" ]; then
    tgt_tokenizer="jieba"
    tgt_chunk_sep=""
else
    tgt_tokenizer="space"
    tgt_chunk_sep=" "
fi

# echo all arguments
echo "prompt_type: $prompt_type"
echo "dataset_name: $dataset_name"
echo "job_name: $job_name"
echo "n: $n"
echo "beam: $beam"
echo "acr: $acr"
echo "data_dir: $data_dir"
echo "lang_pair: $lang_pair"
echo "src_lang: $src_lang"
echo "tgt_lang: $tgt_lang"
echo "model_name: $model_name"
echo "ckpt: $ckpt"
echo "sp_map: $sp_map"
echo "src_tokenizer: $src_tokenizer"
echo "tgt_tokenizer: $tgt_tokenizer"
echo "tgt_chunk_sep: $tgt_chunk_sep"

output=${ROOT_PATH}/eval_result/${dataset_name}/${job_name}/${n}_${beam}_${acr}/
mkdir -p $output

bin_file=${SRC_PATH}/eval/eval_conv_prompt.py
special_args=""
if [ "$prompt_type" == "offline" ]; then
    bin_file=${SRC_PATH}/eval/eval_offline_prompt.py
    special_args=""
elif [ "$prompt_type" == "conv" ]; then
    bin_file=${SRC_PATH}/eval/eval_conv_prompt.py
    special_args="--use-sent-boundary --use-ralcp"
else
    echo "Invalid prompt type: $prompt_type"
    exit 1
fi

PYTHONPATH=$SRC_PATH/eval/ python $bin_file \
    --src ${data_dir}/${lang_pair}.src \
    --tgt ${data_dir}/${lang_pair}.tgt \
    --source-lang ${src_lang} \
    --target-lang ${tgt_lang} \
    --base_model_name ${model_name} \
    --lora_path ${ckpt} \
    --read-n-tokens ${n} \
    --cuda 0 \
    --beam ${beam} \
    --acr ${acr} \
    --special-token-map ${sp_map} \
    --src_tokenizer ${src_tokenizer} \
    --tgt_tokenizer ${tgt_tokenizer} \
    --tgt_chunk_sep ${tgt_chunk_sep} \
    --debug \
    --output ${output} ${special_args} > ${output}/${n}_${beam}_${acr}.log