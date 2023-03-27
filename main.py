"""
Using Huggingface ecosystem to fine-tune an LLM.

References used:
- Huggingface official example for training a causal LM: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
- Huggingface course's chapter about training a causal LM from scratch: https://huggingface.co/course/chapter7/6?fw=pt
"""

# %% LOADING MODEL AND TOKENIZER

import os
import math
from pathlib import Path

import datasets, transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

use_peft = False
dry_run = False
push_to_hub = True
from_base_model = False

base_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
model_name = "vldsavelyev/murakami_rugpt3small"
dataset_name = "vldsavelyev/murakami"
if use_peft:
    model_name += "_peft"

token = os.getenv("HUB_TOKEN")
if push_to_hub and not token:
    raise ValueError(
        "push_to_hub is set to True, but HUB_TOKEN environment variable is not set"
    )

if from_base_model:
    print(f"Loading base model {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # * Adjusting tokenizer and generation config*
    #
    # Some examples might be shorter than the context length, so we will pad them 
    # with DataCollatorForLanguageModeling. Since GPT2 was trained on sequences of 
    # identical size, it didn't do any padding, and didn't bother setting `pad_token` 
    # and `padding_side` in the model config. So we need to set them manually.
    #
    # Note that the default value for `padding_side` is "right", which won't work 
    # well for next-token prediction objective of GPT, so we change it to "left".
    #
    # Also note that the exact value for `pad_token` doesn't matter because it will 
    # be masked anyway in the `attention_mask`. It's often recommended to set 
    # identical to `eos_token` in GPT, however I set it to a different value 
    # (`<pad>` which has the value of 0) to avoid this stupid warning here:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1264-L1273
    # that assumes you had left padding, even when you actually had right padding,
    # because your padding token is the same as your `eos_token` in the end of 
    # a senstence.
    #
    tokenizer.pad_token = '<pad>'
    tokenizer.padding_side = 'left'
    if token and push_to_hub:
        tokenizer.push_to_hub(model_name, use_auth_token=token)
else:
    print(f"Loading checkpoint {model_name} from Hub")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"Model parameters: {model.num_parameters():,}")
if use_peft:
    from peft import get_peft_model, LoraConfig, TaskType

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        ),
    )
    print("Parameter-efficient fine tuning trainable parameters:")
    model.print_trainable_parameters()


# %% LOADING DATASET

print(f"Loading dataset from remote repo {dataset_name}")
dataset = load_dataset(dataset_name)

# Wrap novel chapters with BOS and EOS tokens (tokenizer doesn't do that even
# if add_special_tokens is True, see https://github.com/huggingface/transformers/issues/3311)
dataset = dataset.map(
    lambda x: {'text': f'{tokenizer.bos_token}{x["text"]}{tokenizer.eos_token}'}
)


def _tokenize(batch: dict[str, list]):
    ids = tokenizer(
        batch["text"],
        max_length=model.config.n_ctx,
        truncation=True,  # because of the option below, it will chunk
        return_overflowing_tokens=True,  # ...tokens, not trancate
        # we want the chunks to overlap by 20%
        stride=int(model.config.n_ctx * 0.2),
    )['input_ids']
    return {'input_ids': ids}


dataset = dataset.map(
    _tokenize, batched=True, remove_columns=dataset.column_names["train"]
)


# %% SETUP TRAINER

repos_dir = Path(os.getenv("HUB_REPOS")) or Path().resolve()
save_dir = repos_dir / "models" / model_name

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        push_to_hub=push_to_hub and os.getenv("HUB_TOKEN") is not None,
        hub_model_id=model_name,
        hub_token=os.getenv("HUB_TOKEN"),
        report_to=['all'],
        skip_memory_metrics=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        eval_accumulation_steps=20,
        logging_steps=10,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",  # "cosine" OOMs after ~60 steps
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # Batch size >1 causes EOM on standard T4 Colab GPU (which is weird
        # though that 1 batch is just ~5G, whereas 2 batches EOMs with >15G)
        # Anyways, we are applying optimizations recommended here:
        # https://huggingface.co/docs/transformers/perf_train_gpu_one
        # Note that gradient accumulation doesn't help to rescue the batch
        # sizes >=2, however gradient checkpointing helpes a lot, making it
        # take only 3G for 2 batches, so we can even afford a batch size of 8.
        # Alternatively, using optim="adafactor" helps too, though slightly less.
        gradient_checkpointing=True,
        # gradient_accumulation_steps=4,
        fp16=True,  # Can afford this speed-up optimization. Will boost speed 2x,
        # consuming only a bit more memory (in theory it should take 50% more,
        # but in practice it's below 10%).
        ignore_data_skip=True,  # When restarting from a checkpoint, skip to
        # the batch that should be used at this checkpoint, instead of starting
        # from the first batch.
        # torch_compile=True,  # Compiling triggers a torchdynamo error :-(
        # It's only really useful on Ampere GPUs, anyway.
        # auto_find_batch_size=True,  # Could use this flag to find the first batch
        # that doesn't OOM (https://huggingface.co/docs/accelerate/usage_guides/memory),
        # however in practice it's better to manually try different batch sizes in
        # combination with optimizations like gradient checkpointing, etc (see above),
        # find the best combination and hard-code it.
    )
else:
    # For debugging on a CPU.
    training_args = TrainingArguments(
        output_dir=save_dir,
        report_to=[],
        evaluation_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        logging_first_step=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )

# %% GENERATION

# Model's config has `bos_token_id=eos_token_id=50256`, even though the
# tokenizer has bos_token_id=eos_token_id=50257 ("<|endoftext|>") (for 50256, 
# the tokenizer has just a standard Russian word). That's because the tokenizer
# was completely rebuilt during fine-tuning. The original GPT2 tokenizer didn't
# have other special tokens apart from '<|endoftext|>' (50256), but the rebuilt
# first has first 5 tokens corresponding to default BPE special tokens 
# "<pad>", "<s>", "</s>", "<unk>", "<mask>", which are not used at all, and then
# a special token "<|endoftext|> (50257) added in the end, which was actually used.
# That looks like a bug on their side: when they used a pre-trained GPT2, they
# should have preserved the tokenizer. That resulted in the side-effect that only 
# Russian texts are produced when prompting the model with "<|endoftext|>", whereas
# when prompting with any English letters, you can get English texts. Perhaps
# it was a desired side-effect for Sberbank, but just looks inefficient.
#
# So for our Murakami chapters, we should make sure we we wrap them chapters 
# with 50257.
#
# Generation pipeline reads `model.generation_config` for default values, and
# we pass `eos_token_id` and `pad_token_id` to override those. Annoyingly, it also
# prints the broken `model.generation_config` to stdout, which we can't avoid.
# Modifying `model.generation_config` also doesn't work, as it gets re-set
# by `pipeline` on every run. So we'll have to deal with misleading messages.
#
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    device=training_args.device,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1,
    max_length=200,
)


class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if metrics := kwargs.get("metrics"):
            loss = metrics["eval_loss"]
            print(f'Eval loss: {loss:.4f}')
            print(f'Perplexity: {math.exp(loss):.2f}')
        if state.best_metric:
            print(f"Best loss so far: {state.best_metric:.4f}")

        for result in generator([tokenizer.bos_token])[0]:
            print(result["generated_text"])


trainer = Trainer(
    model=model,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    callbacks=[MyCallback],
    args=training_args,
)

# %% TRAIN
if not dry_run:
    trainer.train(resume_from_checkpoint=get_last_checkpoint(save_dir))
    if push_to_hub:
        trainer.save_model()  # also calls push_to_hub
