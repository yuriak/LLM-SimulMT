import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
import pickle
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import sys
import os
import random
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from transformers import (
    TrainerCallback,
)
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset as Dataset
import tqdm
from torch.utils.data import Dataset as Dset

from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import math
import copy
from torch.nn.utils.rnn import pad_sequence
import mosestokenizer
import re


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


@dataclass
class ScriptArguments:
    source_lang: str = field(metadata={"help": "source language name"})
    target_lang: str = field(metadata={"help": "target language name"})
    data_path: str = field(metadata={"help": "Path to the training data."})
    output_path: str = field(metadata={"help": "Path to save ckpt"})
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    lora_r: int = field(metadata={"help": "lora r"}, default=64)
    lora_alpha: int = field(metadata={"help": "lora alpha"}, default=16)
    turn_length: str = field(metadata={"help": "turn length"}, default="random:2:10")
    offset_p: float = field(metadata={"help": "offset prob"}, default=0.5)
    length_p: float = field(metadata={"help": "length p"}, default=0.6)
    use_offset: bool = field(metadata={"help": "use offset"}, default=False)
    use_merge: bool = field(metadata={"help": "use merge"}, default=False)
    epochs: Optional[int] = field(default=1, metadata={"help": "Epochs for training"})
    steps: Optional[int] = field(
        default=-1, metadata={"help": "max steps for training"}
    )
    bsz: Optional[int] = field(metadata={"help": "batch size"}, default=48)
    grad_accum: Optional[int] = field(
        metadata={"help": "gradient accumulation"}, default=2
    )
    data_scale: Optional[int] = field(
        metadata={"help": "Scale of the training data per language"}, default=99999999
    )
    quant: Optional[bool] = field(metadata={"help": "quant with 4 bit"}, default=False)
    fa2: Optional[bool] = field(
        metadata={"help": "use flash_attention_2"}, default=False
    )
    gradient_checkpointing: Optional[bool] = field(
        metadata={"help": "use gradient checkpoint"}, default=False
    )
    job_name: str = field(metadata={"help": "job name for wandb"}, default=None)
    max_seq_length: int = field(metadata={"help": "max sequence length"}, default=384)
    resume_from_checkpoint: bool = field(
        metadata={"help": "resume_from_checkpoint"}, default=False
    )
    use_sent_boundary: bool = field(
        metadata={"help": "use [B] and [E] as sentence boundary"}, default=False
    )
    logging_steps: int = field(metadata={"help": "logging steps"}, default=100)
    save_on: str = field(metadata={"help": "save on epochs or steps"}, default="steps")
    save_steps: int = field(metadata={"help": "save steps"}, default=10000)
    report_to: str = field(metadata={"help": "wandb or tensorboard"}, default="wandb")
    src_detokenizer: str = field(
        metadata={"help": "detokenizer for source text"}, default=None
    )
    tgt_detokenizer: str = field(
        metadata={"help": "detokenizer for target text"}, default=None
    )
    special_token_map: str = field(
        metadata={"help": "special token map"}, default="llama2"
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

job_name = script_args.job_name

# The model that you want to train from the Hugging Face hub
model_name = script_args.model_name
output_dir = script_args.output_path
data_path = script_args.data_path
data_scale = script_args.data_scale
resume_from_checkpoint = script_args.resume_from_checkpoint
save_on = script_args.save_on

source_lang = script_args.source_lang
target_lang = script_args.target_lang
lora_r = script_args.lora_r
lora_alpha = script_args.lora_alpha
lora_dropout = 0.1
use_4bit = script_args.quant
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
num_train_epochs = script_args.epochs
num_train_steps = script_args.steps
fa2 = script_args.fa2
fp16 = False
bf16 = True
per_device_train_batch_size = script_args.bsz
per_device_eval_batch_size = script_args.bsz
gradient_accumulation_steps = script_args.grad_accum
gradient_checkpointing = script_args.gradient_checkpointing
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
group_by_length = False
save_steps = script_args.save_steps
logging_steps = script_args.logging_steps
max_seq_length = script_args.max_seq_length
packing = False
special_token_map = script_args.special_token_map

turn_length = script_args.turn_length
if "random" in script_args.turn_length:
    min_continuous_read = int(turn_length.split(":")[1])
    max_continuous_read = int(turn_length.split(":")[2])
else:
    min_continuous_read = int(turn_length)
    max_continuous_read = int(turn_length)

use_offset = script_args.use_offset
use_merge = script_args.use_merge
length_p = script_args.length_p
min_length = math.ceil(1 / length_p)
offset_p = script_args.offset_p
use_sent_boundary = script_args.use_sent_boundary

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, add_bos_token=False, add_eos_token=False
)
if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


def zh_detokenize(inputs):
    def is_english_or_number(char):
        """Check if a character is an English letter or a number."""
        return re.match(r"[A-Za-z0-9]", char) is not None

    def is_chinese_char(char):
        """Check if a character is a Chinese character."""
        return "\u4e00" <= char <= "\u9fff"

    def should_add_space(prev_char, current_char):
        """
        Determine if a space should be added between two characters.
        """
        if not prev_char or not current_char:
            return False

        # Add space between English/number and Chinese characters
        if (is_english_or_number(prev_char) and is_chinese_char(current_char)) or (
            is_chinese_char(prev_char) and is_english_or_number(current_char)
        ):
            return True

        # Add space between English words/numbers
        if is_english_or_number(prev_char) and is_english_or_number(current_char):
            return True

        return False

    def detokenize_jieba(tokens):
        """
        Detokenize a list of tokens from Jieba tokenizer into a sentence,
        considering mixed English, Chinese, numbers, and punctuation.

        Parameters:
        tokens (list of str): The list of tokens to be detokenized.

        Returns:
        str: The detokenized sentence.
        """
        sentence = ""
        prev_token = ""
        for token in tokens:
            if prev_token:
                if should_add_space(prev_token[-1], token[0]):
                    sentence += " "
            sentence += token
            prev_token = token

        return sentence

    return detokenize_jieba(inputs)


en_detokenizer = mosestokenizer.MosesDetokenizer(lang="en")

detokenizers = {
    "moses": lambda x: en_detokenizer(
        [c.strip() for c in x.split() if len(c.strip()) > 0]
    ).strip(),
    "zh_detok": lambda x: zh_detokenize(
        [c.strip() for c in x.split() if len(c.strip()) > 0]
    ).strip(),
    "bpe": lambda x: x.replace(" ", "").replace("▁", " ").strip(),
}

src_detokenizer = lambda x: x
tgt_detokenizer = lambda x: x

if script_args.src_detokenizer is not None:
    src_detokenizer = detokenizers.get(script_args.src_detokenizer)
if script_args.tgt_detokenizer is not None:
    tgt_detokenizer = detokenizers.get(script_args.tgt_detokenizer)


def dynamic_merge_seq(seq):
    read_rounds = list(filter(lambda x: x[0] == "R", seq))
    write_rounds = list(filter(lambda x: x[0] == "W", seq))
    assert len(read_rounds) == len(write_rounds)

    all_merged = False
    available_reads = len(read_rounds)
    cursor = 0
    merged_seq = []
    while not all_merged:
        min_read = min(available_reads, min_continuous_read)
        max_read = min(available_reads, max_continuous_read)
        continuous_reads = random.randint(min_read, max_read)
        merged_reads = read_rounds[cursor : cursor + continuous_reads]
        merged_writes = write_rounds[cursor : cursor + continuous_reads]

        merged_seq.append(["R", " ".join([x[1] for x in merged_reads]).strip()])
        merged_seq.append(["W", " ".join([x[1] for x in merged_writes]).strip()])
        cursor += continuous_reads
        available_reads -= continuous_reads
        if available_reads <= 0:
            all_merged = True
    return merged_seq


def make_offset(rw_pairs):
    offseted_rw_pairs = []
    buffer = ""
    rs = rw_pairs[:-1:2]
    ws = rw_pairs[1::2]
    num_pairs = len(rs)
    for i, (r, w) in enumerate(zip(rs, ws)):
        content = w[1]
        if len(buffer) > 0:
            content = " ".join([buffer, content])
            buffer = ""
        tokens = content.split()
        compute_loss = True
        if (
            len(tokens) > min_length
            and random.random() < offset_p
            and i != (num_pairs - 1)
        ):
            num_content_length = random.randint(min_length, len(tokens))
            if num_content_length < len(tokens):
                buffer = " ".join(tokens[num_content_length:])
                content = " ".join(tokens[:num_content_length])
                compute_loss = False
        offseted_rw_pairs.append([r[1], content, compute_loss])
    return offseted_rw_pairs


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


def promptify_n_tokenize(sample, special_tokens_map=LLAMA2_SPECIAL_TOKENS):
    IGNORE_IDX = -100

    system_prompt_template = "{sys_start}Perform simultaneous translation from {source_lang} to {target_lang} through multi-turn dialogue. {sys_end}"
    src_template = "{usr_start}{text}{usr_end}"
    tgt_template = "{ast_start}{text}{ast_end}"
    system_prompt = system_prompt_template.format(
        sys_start=special_tokens_map["sys_start"],
        source_lang=source_lang,
        target_lang=target_lang,
        sys_end=special_tokens_map["sys_end"],
    )
    prompt_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    initial_loss_masks = [IGNORE_IDX] * len(prompt_tokens)

    trajectory = sample["text"]
    if use_merge:
        trajectory = dynamic_merge_seq(trajectory)

    if use_offset:
        trajectory = make_offset(trajectory)
    else:
        trajectory = list(
            zip(
                list(map(lambda x: x[1], trajectory[:-1:2])),
                list(map(lambda x: x[1], trajectory[1::2])),
                [True] * (len(trajectory) // 2),
            )
        )
    all_tokens = copy.deepcopy(prompt_tokens)
    labels = copy.deepcopy(initial_loss_masks)
    for i, (s, t, c_loss) in enumerate(trajectory):
        src_text = src_detokenizer(s)
        tgt_text = tgt_detokenizer(t)

        if use_sent_boundary:
            if i == 0:
                src_text = f"[B] {src_text}"
            elif i == (len(trajectory) - 1):
                src_text = f"{src_text} [E]"
            else:
                pass

        user_prompt = src_template.format(
            usr_start=(
                special_tokens_map["sys_user_start"]
                if i == 0
                else special_tokens_map["usr_start"]
            ),
            text=src_text,
            usr_end=special_tokens_map["usr_end"],
        )
        assistant_prompt = tgt_template.format(
            ast_start=special_tokens_map["ast_start"],
            text=tgt_text,
            ast_end=special_tokens_map["ast_end"],
        )

        stoken = tokenizer.encode(user_prompt)
        ttoken = tokenizer.encode(assistant_prompt)

        turn_labels = [IGNORE_IDX] * len(stoken)

        if c_loss:
            turn_labels += ttoken
        else:
            turn_labels += [IGNORE_IDX] * len(ttoken)

        turn_tokens = stoken + ttoken

        all_tokens += turn_tokens
        labels += turn_labels

    over_length = len(all_tokens) > max_seq_length
    attention_mask = [1] * len(all_tokens)

    if over_length:
        warnings.warn(f"Text over length: {tokenizer.decode(all_tokens)}")
        all_tokens = all_tokens[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = [IGNORE_IDX] * len(attention_mask)

    return {"input_ids": all_tokens, "attention_mask": attention_mask, "labels": labels}


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class SimulDataset(Dset):
    def __init__(self, data_path, tokenizer, shuffle=True):
        super(Dset).__init__()
        self.data = pickle.load(open(data_path, "rb"))

        if data_scale < len(self.data):
            if shuffle:
                random.shuffle(self.data)
            self.data = self.data[:data_scale]

        self.tokenizer = tokenizer
        print("Loaded:", len(self.data))
        self.special_tokens_map = {
            "llama2": LLAMA2_SPECIAL_TOKENS,
            "llama3": LLAMA3_SPECIAL_TOKENS,
            "qwen": QWEN_SPECIAL_TOKENS,
        }.get(special_token_map, LLAMA2_SPECIAL_TOKENS)

    def __getitem__(self, index):
        item = self.data[index]
        model_inputs = promptify_n_tokenize(item, self.special_tokens_map)
        return model_inputs

    def __len__(self):
        return len(self.data)


@dataclass
class DataCollatorForConvPrompt:

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = tokenizer.pad(
            [
                {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
                for x in examples
            ],
            return_tensors="pt",
        )

        labels = pad_sequence(
            [torch.LongTensor(x["labels"]) for x in examples],
            batch_first=True,
            padding_value=-100,
        )
        batch["labels"] = labels
        return batch


def train():

    train_dataset = SimulDataset(os.path.join(data_path, "train_all.pkl"), tokenizer)
    valid_dataset = SimulDataset(os.path.join(data_path, "valid_all.pkl"), tokenizer)

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

    print(Accelerator().local_process_index)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print(model.hf_device_map)

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
    )

    if gradient_checkpointing:
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            preprare_model_kwargs = {
                "use_gradient_checkpointing": gradient_checkpointing
            }
            preprare_model_kwargs["gradient_checkpointing_kwargs"] = {
                "use_reentrant": False
            }
            model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    callbacks = [PeftSavingCallback]

    # Set training parameters
    training_arguments = TrainingArguments(
        run_name=job_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=num_train_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy=save_on,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        save_steps=save_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        save_total_limit=5,
        prediction_loss_only=True,
        # do_eval=True,
        do_train=True,
        log_level="debug",
    )

    data_collator = DataCollatorForConvPrompt()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.model.save_pretrained(output_dir)
    print("Done")


if __name__ == "__main__":
    train()
