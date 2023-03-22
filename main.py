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
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, load_dataset
from huggingface_hub import Repository, create_repo
import fire
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()


class MurakamiDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: np.memmap, n_ctx: int):
        self.token_ids = token_ids
        self.n_ctx = n_ctx

    def __getitem__(self, idx):
        t = torch.LongTensor(self.token_ids[idx : idx + self.n_ctx])
        return {"input_ids": t, "labels": t}

    def __len__(self):
        return len(self.token_ids) - self.n_ctx + 1

    @staticmethod
    def load(
        tokenizer,
        n_ctx: int,
        split: str,
        text_path=None,
        model_name=None,
        repo=None,
        max_n_examples: int = None,
    ) -> "MurakamiDataset":
        if text_path and (pt_path := text_path.with_suffix(".token_ids.pt")).exists():
            print(f"Loading dataset from {pt_path}")
            ids = torch.load(str(pt_path))
        else:
            text = None
            if repo:
                try:
                    d = load_dataset(model_name, split=split, cache_dir=repo.local_dir)
                except:
                    pass
                else:
                    print(f"Loaded text from local dataset repo clone {repo.local_dir}")
                    text = d["text"]
            else:
                try:
                    d = load_dataset(model_name, split=split)
                except:
                    pass
                else:
                    print(f"Loaded text from remote dataset repo {model_name}")
                    text = d["text"]
            if not text:
                print(
                    f"Dataset {model_name} not found on Hub, loading from file {text_path}"
                )
                with open(text_path, "r") as f:
                    text = f.read()
                if token := os.getenv("HUB_TOKEN"):
                    print(f"Pushing to HuggingFace Hub as {split} split")
                    Dataset.from_text(str(text_path), split=split).push_to_hub(
                        model_name, token=token
                    )
                    if repo:
                        print("Syncing local repo after updating remote")
                        repo.git_pull()
            print(f"Characters in text: {len(text):,}")
            ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze().long()
            if max_n_examples:
                max_tokens = max_n_examples + n_ctx - 1
                print(
                    f"Taking first {max_tokens} tokens to make it {max_n_examples} examples"
                )
                ids = ids[:max_tokens]
            eos = torch.tensor([tokenizer.eos_token_id]).long()
            ids = torch.concat((ids, eos))
            torch.save(ids, pt_path)
        print(f"Dataset shape: {ids.shape}")
        return MurakamiDataset(ids, n_ctx)


def main(use_peft=False, dry_run=False, push_to_hub=False):
    base_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    model_name = "vldsavelyev/murakami_rugpt3small"
    if use_peft:
        model_name += "_peft"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    repos_dir = Path(os.getenv("HUGGINGFACE_HUB_REPOS") or "huggingface-hub")
    repo: Repository = None
    if token := os.getenv("HUB_TOKEN"):
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
                f"Finetuned model {model_name} not found, loading from base model {base_model_name}"
            )
            model = AutoModelForCausalLM.from_pretrained(base_model_name)

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

    if token := os.getenv("HUB_TOKEN"):
        create_repo(model_name, token=token, repo_type="dataset", exist_ok=True)
        data_repo = Repository(
            local_dir=repos_dir / "datasets" / model_name,
            clone_from=model_name,
            repo_type="dataset",
        )
        data_repo.git_pull()
        data_dir = None
    else:
        data_dir = "data"

    

    train_set = MurakamiDataset.load(
        tokenizer,
        model.config.n_ctx,
        split="train",
        model_name=model_name,
        repo=data_repo,
        text_path=data_dir / "murakami_train.txt" if data_dir else None,
    )
    test_set = MurakamiDataset.load(
        tokenizer,
        model.config.n_ctx,
        max_n_examples=100,
        split="test",
        model_name=model_name,
        repo=data_repo,
        text_path=data_dir / "murakami_test.txt" if data_dir else None,
    )

    def sample(num_seqs=2, max_length=100):
        set_seed(42)
        for i, seq in enumerate(
            model.generate(
                max_length=max_length,
                top_p=0.95,
                num_return_sequences=num_seqs,
                do_sample=True,
                top_k=50,
                pad_token_id=0,
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

    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=test_set,
        callbacks=[MyCallback],
        args=TrainingArguments(
            output_dir=str(hub_repo_dir),
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
            torch_compile=False,
            # https://huggingface.co/docs/accelerate/usage_guides/memory
            auto_find_batch_size=True,
            fp16=True if torch.cuda.is_available() else False,
        ),
    )
    if not dry_run:
        trainer.train(resume_from_checkpoint=get_last_checkpoint(hub_repo_dir))

        if push_to_hub:
            # Save and push to hub
            trainer.save_model()


if __name__ == "__main__":
    fire.Fire(main)
