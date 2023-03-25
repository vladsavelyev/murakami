import os
from pathlib import Path

import datasets, transformers, evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

use_peft = False
dry_run = False
push_to_hub = True

base_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
model_name = "vldsavelyev/murakami_rugpt3small"
dataset_name = "vldsavelyev/murakami"
if use_peft:
    model_name += "_peft"

repos_dir = Path(os.getenv("HUB_REPOS") or "huggingface-hub")
repo: Repository = None
if token := os.getenv("HUB_TOKEN"):
    print(f"Hub token found, cloning model repo {model_name} to {repos_dir}")
    create_repo(model_name, token=token, exist_ok=True)
    repo = Repository(
        local_dir=repos_dir / "models" / model_name,
        clone_from=model_name,
        token=token,
    )
    repo.git_pull()
    model = AutoModelForCausalLM.from_pretrained(repo.local_dir, local_files_only=True)
else:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        print(
            f"Finetuned model {model_name} not found, loading base model {base_model_name}"
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        print(f"Loaded base model {base_model_name}")
    else:
        print(f"Loaded finetuned model {model_name}")

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

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Some examples might be shorter than the context length, so we will need to pad
# them with DataCollatorForLanguageModeling. Since GPT2 was trained on sequences of the
# same size, it didn't do any padding, and doesn't specify `pad_token` and `padding_side`
# in the config. So we need to set them manually. Note that the exact value for pad_token
# doesn't matter because it will be masked anyway in the `attention_mask`; also note that
# the default value for `padding_side` is right, which won't work well for next-token
# prediction objective of GPT, so we set it to left.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

if token := os.getenv("HUB_TOKEN"):
    print(f"Hub token found, cloning dataset repo {dataset_name} to {repos_dir}")
    create_repo(model_name, token=token, repo_type="dataset", exist_ok=True)
    data_repo = Repository(
        local_dir=repos_dir / "datasets" / dataset_name,
        clone_from=dataset_name,
        repo_type="dataset",
        token=token,
    )
    data_repo.git_pull()
    print(f"Loading dataset from local repo clone at {data_repo.local_dir}")
    dataset = load_dataset(data_repo.local_dir)
else:
    print(f"Loading dataset from remote repo {dataset_name}")
    dataset = load_dataset(dataset_name)

# Following this example to prepare examples:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py


def _tokenize(x):
    return {
        'input_ids': [tokenizer.bos_token_id]
        + tokenizer(x["text"])['input_ids']
        + [tokenizer.eos_token_id]
    }


dataset = dataset.map(_tokenize, batched=False, remove_columns=["text"])


def _chunk(batch: dict[str, list]):
    block_size = model.config.n_ctx
    step = int(block_size * 0.8)  # we want the blocks to overlap by 20%

    chunked_batch = {}
    for k, examples in batch.items():
        chunked_batch[k] = []
        for x in examples:
            for i in range(0, len(x) - block_size + 1, step):
                chunked_batch[k].append(x[i : i + block_size])
            if chunked_batch[k][-1] != x[-1]:
                # if the last chunk containing eos was shorter than the blocks size
                # and wasn't included, we add it explicitly:
                chunked_batch[k].append(x[-block_size:])
    return chunked_batch


dataset = dataset.map(_chunk, batched=True)


def sample(num_seqs=1, max_length=200):
    set_seed(42)
    for i, seq in enumerate(
        model.generate(
            max_length=max_length,
            top_p=0.95,
            num_return_sequences=num_seqs,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    ):
        print(i + 1, tokenizer.decode(seq))


class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if metrics := kwargs.get("metrics"):
            print(f'Eval loss so far: {metrics["eval_loss"]:.4f}')
        if state.best_metric:
            print(f"Best loss so far: {state.best_metric:.4f}")
        sample()


save_dir = str(repo.local_dir) if repo else model_name

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        push_to_hub=push_to_hub and os.getenv("HUB_TOKEN") is not None,
        hub_model_id=model_name,
        hub_token=os.getenv("HUB_TOKEN"),
        skip_memory_metrics=False,
        evaluation_strategy="epochs",
        save_strategy="epochs",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
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
        evaluation_strategy="steps",
        eval_steps=5,
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )

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
if not dry_run:
    trainer.train(resume_from_checkpoint=get_last_checkpoint(save_dir))

    if push_to_hub:
        # Save and push to hub
        trainer.save_model()


repo.push_to_hub(commit_message="End of training 3 epochs")
