# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
from operator import add
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


class MTurnLcpAgent(Agent):
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
        self.use_sent_boundary = args.use_sent_boundary

        self.read_n_tokens = int(args.read_n_tokens)
        if self.read_n_tokens < 0:
            print("Invalid read policy")
            sys.exit(1)
        self.read_count = 1
        self.no_space_punct = (",", ".", ";", "?", "!")
        self.debug = args.debug
        self.offline_model = self.read_n_tokens > 100
        print("Offline model", self.offline_model)

        self.special_tokens_map = {
            "llama2": LLAMA2_SPECIAL_TOKENS,
            "llama3": LLAMA3_SPECIAL_TOKENS,
            "qwen": QWEN_SPECIAL_TOKENS,
        }.get(args.special_token_map, LLAMA2_SPECIAL_TOKENS)

        # Setup special tokens
        self.input_template = "{sys_start}Perform simultaneous translation from {source_lang} to {target_lang} through multi-turn dialogue. {sys_end}".format(
            sys_start=self.special_tokens_map["sys_start"],
            sys_end=self.special_tokens_map["sys_end"],
            source_lang=self.src_lang,
            target_lang=self.tgt_lang,
        )

        self.src_template = "{usr_start}{text}{usr_end}"
        self.tgt_start_template = "{ast_start}"

        self.ast_end_length = len(
            self.tokenizer.encode(
                self.special_tokens_map["ast_end"],
                add_special_tokens=False,
            )
        )

        self.input_template = self.to_device(
            self.tokenizer(
                self.input_template, return_tensors="pt", add_special_tokens=False
            )
        )

        # Setupt generation config
        self.generation_config = self.model.generation_config
        self.generation_config.num_beams = self.beam
        self.generation_config.max_length = 2048
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.max_new_tokens = 256 if not self.offline_model else 1024
        self.generation_config.return_dict_in_generate = True
        self.generation_config.do_sample = False
        self.generation_config.num_return_sequences = self.beam
        self.generation_config.output_scores = True

        self.eos_tokens = self.tokenizer.convert_ids_to_tokens(
            self.generation_config.eos_token_id
        )

        self.acr = args.acr
        self.use_ralcp = args.use_ralcp
        self.tgt_tokenize = lambda x: ' '.join(jieba.lcut(x.replace(" ","▁"))).replace("▁ ","▁")
        self.tgt_detokenize = lambda x: x.replace(" ","").replace("▁"," ")
        self.post_process_zh = lambda x: remove_space_between_chinese_and_punctuation(x)

        self.model.eval()
        self.reset()

    def reset(self) -> None:
        super().reset()
        torch.cuda.empty_cache()
        self.read_count = 1
        self.past_key_values = None
        self.cumulative_sequence = copy.deepcopy(self.input_template["input_ids"]).to(
            self.cuda
        )

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add customized command line arguments"""
        parser.add_argument(
            "--base_model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
        )
        parser.add_argument(
            "--lora_path", type=str, default="llama2_sft_mturn_mustc_1eph"
        )
        parser.add_argument("--source-lang", type=str, default="German")
        parser.add_argument("--target-lang", type=str, default="English")
        parser.add_argument("--read-n-tokens", type=int, default=3)
        parser.add_argument("--cuda", type=int, default=0)
        parser.add_argument("--quant", type=str, default=None)
        parser.add_argument("--beam", type=int, default=5)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--use-sent-boundary", action="store_true")
        parser.add_argument("--acr", type=float, default=1.0)
        parser.add_argument("--tie-breaking", type=int, default=0)
        parser.add_argument("--use-ralcp", action="store_true")
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
        src_chunk = " ".join(self.states.source[-self.read_count :]).strip()
        if source_finished and len(src_chunk) == 0:
            return WriteAction("", finished=source_finished)
        first_round = len(self.states.target) == 0
        if first_round and self.use_sent_boundary:
            src_chunk = "[B] " + src_chunk
        if source_finished and self.use_sent_boundary:
            src_chunk = src_chunk + " [E]"
        if self.debug:
            print("Source states:", self.states.source)
            print("#" * 10)
            print("Source chunk:", src_chunk)
            print("#" * 10)
            print("Target states:", self.states.target)
            print("#" * 10)
        self.read_count = 1

        src_chunk = self.src_template.format(
            usr_start=(
                self.special_tokens_map["usr_start"]
                if not first_round
                else self.special_tokens_map["sys_user_start"]
            ),
            usr_end=self.special_tokens_map["usr_end"],
            text=src_chunk,
        ) + self.tgt_start_template.format(
            ast_start=self.special_tokens_map["ast_start"]
        )

        model_inputs = self.to_device(
            self.tokenizer(src_chunk, return_tensors="pt", add_special_tokens=False)
        )
        if self.debug:
            print("Partial source:", src_chunk)
            print("#" * 10)
        # print(model_inputs)
        start_idx = int(first_round)
        model_inputs = {
            "input_ids": torch.concat(
                [self.cumulative_sequence, model_inputs["input_ids"]], -1
            ),
            "attention_mask": torch.concat(
                [
                    torch.ones_like(self.cumulative_sequence),
                    model_inputs["attention_mask"],
                ],
                -1,
            ),
        }
        if self.past_key_values is not None:
            model_inputs["past_key_values"] = self.past_key_values

        def remove_eos(text):
            if isinstance(self.generation_config.eos_token_id, list):

                for x in self.eos_tokens:
                    text = text.replace(x, "")
            else:
                text = text.replace(
                    self.tokenizer.decode(self.generation_config.eos_token_id), ""
                )
            return text

        if self.beam > 1:

            outputs = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
            )

            beam_outputs = outputs.sequences
            pkv = outputs.past_key_values
            scores = outputs.sequences_scores.exp().tolist()
            input_ids = model_inputs["input_ids"]

            if self.use_ralcp:
                # Apply LCP here
                candidates = [
                    remove_eos(
                        self.tokenizer.decode(
                            x.masked_select(x.ne(self.generation_config.pad_token_id))
                        ).replace(self.special_tokens_map["ast_end"], "")
                    ).strip()
                    for x in beam_outputs[:, input_ids.shape[1] :]
                ]

                if self.debug:
                    print("#" * 10)
                    print("Candidates:", candidates)
                    print("Scores:", scores)
                prefix, idx = self.policy_str(
                    candidates,
                    scores,
                    src_finished=source_finished,
                    accept_ratio=self.acr,
                    tie_breaking=-1,
                    weighted=False,
                )
                if self.debug:
                    print("#" * 10)
                    print("Prefix:", prefix)
                prefix_len = len(
                    self.tokenizer.tokenize(prefix, add_special_tokens=False)
                )
                preserved_prefix_len = input_ids.shape[1] + prefix_len
                pkv.crop(preserved_prefix_len)
                pkv.batch_select_indices([idx] * self.beam)

                self.past_key_values = pkv

                self.cumulative_sequence = torch.concat(
                    [
                        beam_outputs[idx, :preserved_prefix_len],
                        torch.LongTensor(
                            self.tokenizer.encode(
                                self.special_tokens_map["ast_end"],
                                add_special_tokens=False,
                            )
                        ).to(beam_outputs.device),
                    ]
                ).unsqueeze(0)
                response = prefix
            else:

                # Get the first beam output, remove all eos, as the prefix.
                prefix = remove_eos(
                    self.tokenizer.decode(
                        beam_outputs[0, input_ids.shape[1] :]
                    ).replace(self.special_tokens_map["ast_end"], "")
                ).strip()
                prefix_len = len(
                    self.tokenizer.tokenize(prefix, add_special_tokens=False)
                )
                pkv.batch_select_indices([0] * self.beam)
                pkv.crop(input_ids.shape[1] + prefix_len)
                self.past_key_values = pkv
                # Rebuild cumulative sequence here, add the standard ast_end token
                self.cumulative_sequence = torch.concat(
                    [
                        beam_outputs[0, : input_ids.shape[1] + prefix_len],
                        torch.LongTensor(
                            self.tokenizer.encode(
                                self.special_tokens_map["ast_end"],
                                add_special_tokens=False,
                            )
                        ).to(beam_outputs.device),
                    ]
                ).unsqueeze(0)
                # Get the latest result
                response = prefix

        else:
            # Greedy search

            outputs = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
            )

            greedy_outputs = outputs.sequences
            pkv = outputs.past_key_values
            input_ids = model_inputs["input_ids"]
            self.cumulative_sequence = greedy_outputs
            # remove all eos, as the prefix.
            prefix = remove_eos(
                self.tokenizer.decode(greedy_outputs[0, input_ids.shape[1] :]).replace(
                    self.special_tokens_map["ast_end"], ""
                )
            ).strip()
            prefix_len = len(self.tokenizer.tokenize(prefix, add_special_tokens=False))
            pkv.crop(input_ids.shape[1] + prefix_len)
            pkv.batch_select_indices([0] * self.beam)
            self.past_key_values = pkv
            # Rebuild cumulative sequence here, add the standard ast_end token
            self.cumulative_sequence = torch.concat(
                [
                    greedy_outputs[0, : input_ids.shape[1] + prefix_len],
                    torch.LongTensor(
                        self.tokenizer.encode(
                            self.special_tokens_map["ast_end"],
                            add_special_tokens=False,
                        )
                    ).to(greedy_outputs.device),
                ]
            ).unsqueeze(0)
            response = prefix

        if self.debug:
            print(
                "Cumulative response",
                self.tokenizer.decode(self.cumulative_sequence[0]),
            )
            print("#" * 10)
            print("Response", response)
            print("#" * 10)
            print("Src finished:", source_finished)
            print("---" * 10)
        if self.tgt_lang == "Chinese":
            response = self.post_process_zh(response)
        return WriteAction(response, finished=source_finished)

    def policy_str(
        self,
        out_seq_raw,
        scores,
        src_finished=False,
        accept_ratio=0.6,
        tie_breaking=-1,
        weighted=True,
    ):
        preserved_prefix = []

        selected_path = 0

        def score(ts, ss):
            if weighted:
                result = {t: 0 for t in ts}
                for t, s in zip(ts, ss):
                    result[t] += s
                # print(result)

                return OrderedDict(list(sorted(result.items(), key=lambda x: x[1])))
            else:
                return Counter(ts)
        if self.tgt_lang == "Chinese":
            out_seq = list(map(lambda x: self.tgt_tokenize(x), out_seq_raw))
        else:
            out_seq = out_seq_raw
        if not src_finished:
            alive_track_id = set()
            pruned_track = set()
            for i, x in enumerate(out_seq):
                if len(x.strip()) == 0:
                    pruned_track.add(i)
                else:
                    alive_track_id.add(i)

            pos = 0
            while len(alive_track_id) > 1:
                alive_tokens = [
                    out_seq[i].split() for i in sorted(list(alive_track_id))
                ]
                remaining_candidates = [
                    (aid, tokens[pos])
                    for aid, tokens in zip(alive_track_id, alive_tokens)
                    if pos < len(tokens)
                ]
                if self.debug:
                    print("Remaining", remaining_candidates)
                if len(remaining_candidates) > 1:
                    if len(set(list(map(lambda x: x[1], remaining_candidates)))) == 1:
                        selected_path = remaining_candidates[0][0]
                        preserved_prefix.append(remaining_candidates[0][1])
                    else:
                        risky_add = False
                        max_k, max_v = None, 0.0
                        rank_result = score(
                            list(map(lambda x: x[1], remaining_candidates)), None
                        )
                        for k, v in rank_result.items():
                            if self.debug:
                                print(
                                    "token:",
                                    k,
                                    "number:",
                                    v,
                                    "min_threshold:",
                                    len(remaining_candidates) * accept_ratio,
                                )
                                print("max_v:", max_v, "max_k:", max_k, v > max_v)
                            if v > max_v:
                                max_v = v
                                max_k = k
                            if v >= len(remaining_candidates) * accept_ratio:
                                risky_add = True
                                preserved_prefix.append(k)
                                selected_path = [
                                    aid
                                    for aid, token in remaining_candidates
                                    if token == k
                                ][0]
                                if self.debug:
                                    print("path selected", selected_path)

                                alive_track_id = set(
                                    [
                                        aid
                                        for aid, token in remaining_candidates
                                        if token == k
                                    ]
                                )
                                pruned_track.union(
                                    set(
                                        [
                                            aid
                                            for aid, token in remaining_candidates
                                            if token != k
                                        ]
                                    )
                                )
                                if self.debug:
                                    print("prunded track", pruned_track)

                                break
                        if not risky_add:
                            if pos == 0:
                                # Do something to make sure to have one
                                if tie_breaking == 1:
                                    if self.tgt_lang == "Chinese":
                                        return out_seq_raw[0], 0
                                    return out_seq[0], 0

                                if tie_breaking == -1:
                                    force_selected_paths = list(
                                        filter(
                                            lambda x: x[1] == max_k,
                                            remaining_candidates,
                                        )
                                    )
                                    force_selected_token = force_selected_paths[0][1]
                                    alive_track_id = set(
                                        [x[0] for x in force_selected_paths]
                                    )
                                    if self.debug:
                                        print(
                                            "Force to select agg=-1:",
                                            force_selected_token,
                                        )
                                    preserved_prefix.append(force_selected_token)

                                elif tie_breaking == 0:
                                    force_selected_token = remaining_candidates[0][1]
                                    if self.debug:
                                        print(
                                            "Force to select agg=1:",
                                            force_selected_token,
                                        )
                                    preserved_prefix.append(force_selected_token)
                                    break

                            else:
                                break
                else:
                    break
                pos += 1
        else:
            if self.tgt_lang == "Chinese":
                return out_seq_raw[0], 0
            return out_seq[0], 0
        if self.tgt_lang == "Chinese":
            prefix = self.tgt_detokenize(" ".join(preserved_prefix))
        else:
            prefix = " ".join(preserved_prefix)
        return prefix, selected_path


if __name__ == "__main__":
    parser = ArgumentParser()
    Evaluator.add_args(parser)
    MTurnLcpAgent.add_args(parser)
    args = parser.parse_args()
    agent = MTurnLcpAgent(args)
    evaluator = Evaluator(args, agent)
    results = evaluator.evaluate()
    evaluator.dump_results(results)
    evaluator.compute_performance(results)
