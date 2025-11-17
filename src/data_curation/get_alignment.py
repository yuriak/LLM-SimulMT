import pandas as pd
import numpy as np
import sacrebleu
import tqdm
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file-type", default="csv")
parser.add_argument("--split", default="all")
parser.add_argument("--input-path", default='./')
parser.add_argument("--output-path", default='./trajectory')
parser.add_argument("--fastalign-path", default='./tools/fast_align/')
parser.add_argument("--lang", help="in <src>-<tgt>")
parser.add_argument("--reverse-lang", help="reverse src and tgt in the orig file", action="store_true")
args = parser.parse_args()

READ_FUNC = {
    'json': pd.read_json,
    'csv': pd.read_csv
}

clean_text = lambda s: [str(x).strip() for x in s]


def learn_alignment(lang, file_type='csv', input_path='./', output_path='./', split='all', reverse=False):
    src, tgt = lang.split('-')

    src_key = "src" if not reverse else "tgt"
    tgt_key = "tgt" if not reverse else "src"
    if split == 'all':
        read = READ_FUNC[file_type]
        train_df = read(f'{input_path}/train.{file_type}')
        dev_df = read(f'{input_path}/valid.{file_type}')
        open(f'{output_path}/{lang}.train.{src}', 'w').write('\n'.join(clean_text(train_df[src_key].tolist())) + '\n')
        open(f'{output_path}/{lang}.train.{tgt}', 'w').write('\n'.join(clean_text(train_df[tgt_key].tolist())) + '\n')
        open(f'{output_path}/{lang}.valid.{src}', 'w').write('\n'.join(clean_text(dev_df[src_key].tolist())) + '\n')
        open(f'{output_path}/{lang}.valid.{tgt}', 'w').write('\n'.join(clean_text(dev_df[tgt_key].tolist())) + '\n')
        df_sizes = [train_df.shape[0], dev_df.shape[0]]
        df = pd.concat([train_df, dev_df], axis=0)
        open(f'{output_path}/{lang}.{src}', 'w').write('\n'.join(clean_text(df[src_key].tolist())) + '\n')
        open(f'{output_path}/{lang}.{tgt}', 'w').write('\n'.join(clean_text(df[tgt_key].tolist())) + '\n')
    else:
        df = pd.read_csv(f'{input_path}/{split}.csv')
        df_sizes = [df.shape[0]]
        open(f'{output_path}/{lang}.{src}', 'w').write('\n'.join(clean_text(df[src_key].tolist())) + '\n')
        open(f'{output_path}/{lang}.{tgt}', 'w').write('\n'.join(clean_text(df[tgt_key].tolist())) + '\n')
    awk = "awk 'FNR==NR {a[FNR]=$0; next} {print a[FNR] \" ||| \" $0}' " + f"{output_path}/{lang}.{src} {output_path}/{lang}.{tgt} > {output_path}/{lang}.nontok"
    os.system(f"{awk}")
    os.system(
        f"{args.fastalign_path}/build/fast_align -i {output_path}/{lang}.nontok -d -o -v > {output_path}/forward.nontok.{lang}")
    os.system(
        f"{args.fastalign_path}/build/fast_align -i {output_path}/{lang}.nontok -d -o -v -r > {output_path}/reverse.nontok.{lang}")
    os.system(
        f"{args.fastalign_path}/build/atools -i {output_path}/forward.nontok.{lang} -j {output_path}/reverse.nontok.{lang} -c grow-diag-final-and > {output_path}/alignment.nontok.{lang}")
    if split == 'all':
        os.system(
            f"sed -n '1,{df_sizes[0]}p' {output_path}/alignment.nontok.{lang} > {output_path}/alignment.nontok.train.{lang}")
        os.system(
            f"sed -n '1,{df_sizes[0]}p' {output_path}/forward.nontok.{lang} > {output_path}/forward.nontok.train.{lang}")
        os.system(
            f"sed -n '1,{df_sizes[0]}p' {output_path}/reverse.nontok.{lang} > {output_path}/reverse.nontok.train.{lang}")
        os.system(
            f"sed -n '{df_sizes[0] + 1},{df_sizes[0] + df_sizes[1]}p' {output_path}/alignment.nontok.{lang} > {output_path}/alignment.nontok.valid.{lang}")
        os.system(
            f"sed -n '{df_sizes[0] + 1},{df_sizes[0] + df_sizes[1]}p' {output_path}/forward.nontok.{lang} > {output_path}/forward.nontok.valid.{lang}")
        os.system(
            f"sed -n '{df_sizes[0] + 1},{df_sizes[0] + df_sizes[1]}p' {output_path}/reverse.nontok.{lang} > {output_path}/reverse.nontok.valid.{lang}")


os.makedirs(args.output_path, exist_ok=True)
learn_alignment(args.lang, args.file_type, args.input_path, args.output_path, args.split)
