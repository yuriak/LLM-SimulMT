# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random
import sys

sys.path.append("./")

from argparse import Namespace, ArgumentParser
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer,
)
import torch
from peft import PeftModel
from collections import Counter
import string
from simul_evaluator import Evaluator, ReadAction, WriteAction, Agent, AgentStates

from collections import OrderedDict
from collections import Counter
import time
import tqdm
import json
import os
import numpy as np
import jieba
import re

pack_to_device = lambda x, d: {k: v.cuda(d) for k, v in x.items()}


LLAMA3_SPECIAL_TOKENS = {
    "sys_start": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "sys_end": "<|eot_id|>",
    "sys_user_start": "<|start_header_id|>user<|end_header_id|>",
    "usr_start": "<|start_header_id|>user<|end_header_id|>",
    "usr_end": "<|eot_id|>",
    "ast_start": "<|start_header_id|>assistant<|end_header_id|>",
    "ast_end": "<|eot_id|>",
}

LLAMA2_SPECIAL_TOKENS = {
    "sys_start": "<s>[INST] <<SYS>>\n",
    "sys_end": "\n<</SYS>>",
    "sys_user_start": "\n\n",
    "usr_start": "<s>[INST] ",
    "usr_end": " ",
    "ast_start": "[/INST] ",
    "ast_end": " </s>",
}

QWEN_SPECIAL_TOKENS = {
    "sys_start": "<|im_start|>system\n",
    "sys_end": "<|im_end|>\n",
    "sys_user_start": "<|im_start|>user\n",
    "usr_start": "<|im_start|>user\n",
    "usr_end": "<|im_end|>\n",
    "ast_start": "<|im_start|>assistant\n",
    "ast_end": "<|im_end|>\n",
}


def remove_space_between_chinese_and_punctuation(text):
    if not text:
        return ""
    
    chinese_char_pattern = r'[\u4e00-\u9fff]'
    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~\u3000-\u303F\uFF00-\uFFEF]'
    
    result = re.sub(f'({chinese_char_pattern}) +({punctuation_pattern})', r'\1\2', text)
    
    result = re.sub(f'({punctuation_pattern}) +({chinese_char_pattern})', r'\1\2', result)
    
    return result

class OfflineLcpAgent(Agent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """

        super().__init__(args)
        self.cuda = int(args.cuda)
        self.to_device = lambda x: (
            x.cuda(self.cuda)
            if type(x) == torch.Tensor
            else {k: v.cuda(self.cuda) for k, v in x.items()}
        )
        self.quant = args.quant
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.cuda},
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model = model.merge_and_unload()

        self.model = model
        self.model.config.use_cache = True
        self.model.config.use_bfloat16 = True
        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            trust_remote_code=True,
            add_bos_token=False,
            add_eos_token=False,
        )
        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.beam = args.beam
        self.src_lang = args.source_lang
        self.tgt_lang = args.target_lang
        self.acr = args.acr

        self.pipeline = pipeline(
            task="text-generation", model=self.model, tokenizer=self.tokenizer
        )

        self.special_tokens_map = {
            "llama2": LLAMA2_SPECIAL_TOKENS,
            "llama3": LLAMA3_SPECIAL_TOKENS,
            "qwen": QWEN_SPECIAL_TOKENS,
        }.get(args.special_token_map, LLAMA2_SPECIAL_TOKENS)

        self.read_n_tokens = int(args.read_n_tokens)
        if self.read_n_tokens < 0:
            print("Invalid read policy")
            sys.exit(1)
        self.read_count = 1
        self.debug = args.debug
        self.offline_model = self.read_n_tokens > 100
        self.max_token_per_step = 64 if not self.offline_model else 512
        print("Offline model", self.offline_model)
        self.no_space_punct = (",", ".", ";", "?", "!")
        self.tgt_tokenize = lambda x: ' '.join(jieba.lcut(x.replace(" ","▁"))).replace("▁ ","▁")
        self.tgt_detokenize = lambda x: x.replace(" ","").replace("▁"," ")
        self.post_process_zh = lambda x: remove_space_between_chinese_and_punctuation(x)

        self.model.eval()
        self.reset()

        self.system_input_template = (
            "{sys_start}You are a professional translator.{sys_end}".format(
                sys_start=self.special_tokens_map["sys_start"],
                sys_end=self.special_tokens_map["sys_end"],
            )
        )

        self.user_input_template = "{usr_start}{text}{usr_end}"
        self.assistant_input_template = "{ast_start}{text}"

    def reset(self) -> None:
        super().reset()
        self.read_count = 1
        torch.cuda.empty_cache()

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add customized command line arguments"""
        parser.add_argument(
            "--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf"
        )
        parser.add_argument(
            "--lora_path",
            type=str,
            default="llama2_prefix_1000_results_mustc_ende_full",
        )
        parser.add_argument("--source-lang", type=str, default="English")
        parser.add_argument("--target-lang", type=str, default="German")
        parser.add_argument("--read-n-tokens", type=int, default=3)
        parser.add_argument("--cuda", type=int, default=0)
        parser.add_argument("--quant", type=str, default=None)
        parser.add_argument("--beam", type=int, default=5)
        parser.add_argument("--acr", type=float, default=0.6)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--special-token-map", type=str, default="llama2")

    def policy(self):
        source_finished = self.states.source_finished
        if not source_finished:
            if self.debug:
                print("read count", self.read_count)
                print("source state content", self.states.source)
                print("target state content", self.states.target)
                print("#" * 10)
            if self.read_count % self.read_n_tokens != 0:
                self.read_count += 1
                return ReadAction()
        src_chunk = " ".join(self.states.source).strip()
        if source_finished and len(src_chunk) == 0:
            return WriteAction("", finished=source_finished)
        prefix = self.punct_safe_merge(self.states.target)

        user_text = f"Translate the sentence from {self.src_lang} to {self.tgt_lang}:{{{src_chunk}}}"
        assistant_text = f"{{{prefix}"

        model_input = (
            self.system_input_template
            + self.user_input_template.format(
                usr_start=self.special_tokens_map["usr_start"],
                usr_end=self.special_tokens_map["usr_end"],
                text=user_text,
            )
            + self.assistant_input_template.format(
                ast_start=self.special_tokens_map["ast_start"], text=assistant_text
            )
        )

        model_output = self.pipeline(
            model_input,
            num_beams=self.beam,
            num_return_sequences=self.beam,
            do_sample=False,
            max_new_tokens=self.max_token_per_step,
            return_full_text=False,
        )
        trans_out_seq = [x["generated_text"] for x in model_output]

        possible_new_tokens = self.policy_str(
            trans_out_seq,
            src_finished=source_finished,
            beam_size=self.beam,
            accept_ratio=self.acr,
        )
        if self.tgt_lang == "Chinese":
            possible_new_tokens = possible_new_tokens.rstrip()
        else:
            possible_new_tokens = possible_new_tokens.strip()
        if len(possible_new_tokens) > 0 and possible_new_tokens[-1] == "}":
            possible_new_tokens = possible_new_tokens[:-1]

        if self.debug:
            print("Partial source:", src_chunk)
            print("#" * 10)
            print("Input with response", model_input)
            print("#" * 10)
            print("latest translation:", trans_out_seq)
            print("#" * 10)
            print("Src finished:", source_finished)
            print("#" * 10)
            print(
                "Latest decoded:",
                possible_new_tokens,
                "|| Len:",
                len(possible_new_tokens),
            )
            print("---" * 10)
        if len(possible_new_tokens) == 0 and not source_finished:
            self.read_count += 1
            return ReadAction()
        self.read_count = 1
        if self.tgt_lang == "Chinese":
            possible_new_tokens = self.post_process_zh(possible_new_tokens)
        return WriteAction(possible_new_tokens, finished=source_finished)

    def punct_safe_merge(self, states):
        if self.tgt_lang == "Chinese":
            return "".join(states).strip()
        output = ""
        for x in states:
            sep = " " if x[0] not in string.punctuation else ""
            output = output + sep + x
        return output.strip()

    def policy_str(self, out_seq, src_finished=False, beam_size=5, accept_ratio=0.6):
        preserved_prefix = []
        if not src_finished:
            if self.tgt_lang == "Chinese":
                out_seq = list(map(lambda x: self.tgt_tokenize(x), out_seq))
            else:
                out_seq = out_seq
            for candidates in list(zip(*[x.split() for x in out_seq])):
                if len(set(candidates)) == 1:
                    preserved_prefix.append(candidates[0])
                else:
                    risky_add = False
                    for k, v in Counter(candidates).items():
                        if v >= int(beam_size * accept_ratio):
                            preserved_prefix.append(k)
                            risky_add = True
                            break
                    if not risky_add:
                        break
        else:
            finished_target = list(
                filter(lambda x: len(x.strip()) > 0 and x.strip()[-1] == "}", out_seq)
            )
            if len(finished_target) > 0:
                return finished_target[0]
            return out_seq[0]
        if self.tgt_lang == "Chinese":
            return self.tgt_detokenize(" ".join(preserved_prefix))
        else:
            return " ".join(preserved_prefix)


if __name__ == "__main__":
    parser = ArgumentParser()
    Evaluator.add_args(parser)
    OfflineLcpAgent.add_args(parser)
    args = parser.parse_args()
    agent = OfflineLcpAgent(args)
    evaluator = Evaluator(args, agent)
    results = evaluator.evaluate()
    evaluator.dump_results(results)
    evaluator.compute_performance(results)
