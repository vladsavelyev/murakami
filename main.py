import os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset as HFDataset
from datasets import load_dataset
from accelerate.utils import set_seed
import fire
import coloredlogs

coloredlogs.install(level="DEBUG")


source_model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
model_name = "vldsavelyev/murakami_rugpt3small"


class MurakamiDataset(Dataset):
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
        txt_path,
        tokenizer,
        n_ctx: int,
        split: str,
        max_n_examples: int = None,
    ) -> "MurakamiDataset":
        if (pt_path := txt_path.with_suffix(".token_ids.pt")).exists():
            print(f"Loading dataset from {pt_path}")
            ids = torch.load(str(pt_path))
        else:
            text = None
            if token := os.getenv("HUB_TOKEN"):
                try:
                    d = load_dataset(model_name, split=split, use_auth_token=token)
                except:
                    pass
                else:
                    text = d["text"]
            if not text:
                print(f"Dataset {model_name} not found on Hub, loading from file {txt_path}")
                with open(txt_path, "r") as f:
                    text = f.read()
                if token := os.getenv("HUB_TOKEN"):
                    print(f"Pushing to HuggingFace Hub as {split} split")
                    HFDataset.from_text(str(txt_path), split=split).push_to_hub(
                        "vldsavelyev/murakami_rugpt3small", token=os.getenv("HUB_TOKEN")
                    )
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


def main(data_dir="data", use_peft=False, dry_run=False):
    data_dir = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        print(
            f"Model {model_name} not found, loading from source model {source_model_name}"
        )
        model = AutoModelForCausalLM.from_pretrained(source_model_name)
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

    train_set = MurakamiDataset.load(
        data_dir / "murakami_train.txt", tokenizer, model.config.n_ctx, split="train"
    )
    test_set = MurakamiDataset.load(
        data_dir / "murakami_test.txt",
        tokenizer,
        model.config.n_ctx,
        max_n_examples=100,
        split="test",
    )

    save_dir = Path("saves") / f"murakami_rugpt3small{'_peft' if use_peft else ''}"
    save_dir.mkdir(exist_ok=True, parents=True)
    if last_checkpoint_dir := get_last_checkpoint(str(save_dir)):
        last_checkpoint_dir = Path(last_checkpoint_dir)
        print([t.name for t in last_checkpoint_dir.iterdir()])

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
            output_dir=str(save_dir),
            push_to_hub=os.getenv("HUB_TOKEN") is not None,
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
        trainer.train(resume_from_checkpoint=last_checkpoint_dir)

    # Save and push to hub
    trainer.save_model()


if __name__ == "__main__":
    fire.Fire(main)
