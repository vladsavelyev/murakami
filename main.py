import os
from pathlib import Path
import numpy as np

import torch
import datasets, transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    set_seed,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, load_dataset
from huggingface_hub import Repository, create_repo
import fire
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()


def main(use_peft=False, dry_run=False, push_to_hub=False, debugging=False):
    base_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    model_name = "vldsavelyev/murakami_rugpt3small"
    dataset_name = "vldsavelyev/murakami"
    if use_peft:
        model_name += "_peft"

    repos_dir = Path(os.getenv("HUGGINGFACE_HUB_REPOS") or "huggingface-hub")
    repo: Repository = None
    if token := os.getenv("HUB_TOKEN"):
        print(f"Hub token found, cloning model repo {model_name} to {repos_dir}")
        create_repo(model_name, token=token, exist_ok=True)
        repo = Repository(
            local_dir=repos_dir / "models" / model_name, clone_from=model_name
        )
        repo.git_pull()
        model = AutoModelForCausalLM.from_pretrained(
            repo.local_dir, local_files_only=True
        )
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
    # them with DataCollatorForLanguageModeling
    tokenizer.pad_token = tokenizer.eos_token

    if token := os.getenv("HUB_TOKEN"):
        print(f"Hub token found, cloning dataset repo {dataset_name} to {repos_dir}")
        create_repo(model_name, token=token, repo_type="dataset", exist_ok=True)
        data_repo = Repository(
            local_dir=repos_dir / "datasets" / dataset_name,
            clone_from=dataset_name,
            repo_type="dataset",
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
    dataset.flatten()


    def sample(num_seqs=2, max_length=100):
        set_seed(42)
        for i, seq in enumerate(
            model.generate(
                max_length=max_length,
                top_p=0.95,
                num_return_sequences=num_seqs,
                do_sample=True,
                top_k=50,
                # pad_token_id=tokenizer.pad_token_id,
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
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        callbacks=[MyCallback],
        args=TrainingArguments(
            output_dir=save_dir,
            push_to_hub=push_to_hub and os.getenv("HUB_TOKEN") is not None,
            hub_model_id=model_name,
            hub_token=os.getenv("HUB_TOKEN"),
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else None,
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            eval_steps=5000,
            save_steps=5000,
            save_total_limit=2,
            # per_device_train_batch_size=2,
            # per_device_eval_batch_size=2,
            ignore_data_skip=True,
            torch_compile=not debugging,
            # https://huggingface.co/docs/accelerate/usage_guides/memory
            auto_find_batch_size=True,
            skip_memory_metrics=False,
        ),
    )
    mem_b = model.num_parameters() * sum(
        [
            4,  # weights
            8,  # AdamW's two states
            4,  # gradients
        ]
    )
    print("Estimated model GPU memory usage: {:.2f} GB".format(mem_b / 1024**3))

    if not dry_run:
        trainer.train(resume_from_checkpoint=get_last_checkpoint(save_dir))

        if push_to_hub:
            # Save and push to hub
            trainer.save_model()


if __name__ == "__main__":
    fire.Fire(main)
