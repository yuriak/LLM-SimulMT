import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
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
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ScriptArguments:
    source_lang: str = field(metadata={"help": "source language name"})
    target_lang: str = field(metadata={"help": "target language name"})
    data_path: Optional[str] = field(
        metadata={"help": "Path to the training data."}, default=None
    )
    output_path: Optional[str] = field(
        metadata={"help": "Path to save ckpt"}, default=None
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    lora_r: int = field(metadata={"help": "lora r"}, default=64)
    lora_alpha: int = field(metadata={"help": "lora alpha"}, default=16)
    bsz: Optional[int] = field(metadata={"help": "batch size"}, default=48)
    grad_accum: Optional[int] = field(
        metadata={"help": "gradient accumulation"}, default=2
    )
    epochs: Optional[int] = field(default=1, metadata={"help": "Epochs for training"})
    steps: Optional[int] = field(
        default=-1, metadata={"help": "max steps for training"}
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
    logging_steps: int = field(metadata={"help": "logging steps"}, default=100)
    save_on: str = field(metadata={"help": "save on epochs or steps"}, default="steps")
    save_steps: int = field(metadata={"help": "save steps"}, default=10000)
    report_to: str = field(metadata={"help": "wandb or tensorboard"}, default="wandb")
    reverse_lang: bool = field(
        metadata={"help": "whether to reverse src and tgt"}, default=False
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
reverse_lang = script_args.reverse_lang

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
group_by_length = True
save_steps = script_args.save_steps
logging_steps = script_args.logging_steps
max_seq_length = script_args.max_seq_length
packing = False


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

special_token_map = {
    "llama2": LLAMA2_SPECIAL_TOKENS,
    "llama3": LLAMA3_SPECIAL_TOKENS,
    "qwen": QWEN_SPECIAL_TOKENS,
}.get(script_args.special_token_map, LLAMA2_SPECIAL_TOKENS)


def generate_translation_instruction(src_text, src_lang, tgt_lang):
    instructions = [
        f"Translate the following sentence: {{{src_text}}} from {src_lang} to {tgt_lang}.",
        f"I need a translation from {src_lang} to {tgt_lang} for the text: {{{src_text}}}.",
        f"Please translate {{{src_text}}} from {src_lang} to {tgt_lang}.",
        f"Could you help me translate {{{src_text}}} from {src_lang} to {tgt_lang}?",
        f"I require a translation of {{{src_text}}} from {src_lang} to {tgt_lang}.",
        f"Take the sentence {{{src_text}}} in {src_lang} and translate it to {tgt_lang}.",
        f"Translate {{{src_text}}} from {src_lang} to {tgt_lang}.",
        f"Provide me with a translation from {src_lang} to {tgt_lang} for the text: {{{src_text}}}.",
        f"I'm looking for a translation of {{{src_text}}} from {src_lang} to {tgt_lang}.",
        f"Translate the sentence {{{src_text}}} from {src_lang} to {tgt_lang}.",
    ]

    return random.choice(instructions)


system_prompt_template = "{sys_start}You are a professional translator.{sys_end}"
src_template = "{usr_start}{text}{usr_end}"
tgt_template = "{ast_start}{text}{ast_end}"


def transform_dataset(example):
    src_key = "src" if not reverse_lang else "tgt"
    tgt_key = "tgt" if not reverse_lang else "src"
    src_text = example[src_key]
    tgt_text = example[tgt_key]
    instruction = generate_translation_instruction(src_text, source_lang, target_lang)
    tgt_text = f"{{{tgt_text}}}"
    system_prompt = system_prompt_template.format(
        sys_start=special_token_map["sys_start"], sys_end=special_token_map["sys_end"]
    )
    user_input = src_template.format(
        usr_start=special_token_map["usr_start"],
        text=instruction,
        usr_end=special_token_map["usr_end"],
    )
    assistant_output = tgt_template.format(
        ast_start=special_token_map["ast_start"],
        text=tgt_text,
        ast_end=special_token_map["ast_end"],
    )
    llm_text = f"{system_prompt}{user_input}{assistant_output}"

    return {"text": llm_text}


def formatting_prompts_func(example):
    return example["text"]


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class DataCollatorForOfflinePrompt(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        instruction_template (`Optional[str]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Union[str, List[int]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if (
            not self.mlm
            and self.instruction_template
            and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(
                    batch["labels"][i] == self.response_token_ids[0]
                )[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            assistant_idx : assistant_idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_idxs.append(
                            assistant_idx + len(self.response_token_ids)
                        )

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if (
                        human_token_ids
                        == batch["labels"][i][
                            human_idx : human_idx + len(human_token_ids)
                        ].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                for idx, (start, end) in enumerate(
                    zip(human_token_ids_idxs, response_token_ids_idxs)
                ):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch


def train():
    train_dataset = load_dataset(data_path, split="train").shuffle()
    valid_dataset = load_dataset(data_path, split="validation")
    if data_scale < len(train_dataset):
        train_dataset = train_dataset.select(range(data_scale))

    train_dataset = train_dataset.map(transform_dataset, remove_columns=["src", "tgt"])
    valid_dataset = valid_dataset.map(transform_dataset, remove_columns=["src", "tgt"])

    # Load tokenizer and model with QLoRA configuration
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
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print(model.hf_device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, add_bos_token=False, add_eos_token=False
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
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
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
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

    response_template = special_token_map["ast_start"].strip()

    data_collator = DataCollatorForOfflinePrompt(
        response_template, tokenizer=tokenizer, mlm=False
    )

    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=True,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=None,
        batch_size=10000,
    )
    train_dataset = train_dataset.filter(
        lambda x: x["length"] < max_seq_length - 5
    ).remove_columns("length")

    valid_dataset = valid_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=True,
        ),
        batched=True,
        remove_columns=valid_dataset.column_names,
        num_proc=None,
        batch_size=10000,
    )
    valid_dataset = valid_dataset.filter(
        lambda x: x["length"] < max_seq_length - 5
    ).remove_columns("length")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
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
