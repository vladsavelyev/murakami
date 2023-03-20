import os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from accelerate.utils import set_seed
import fire
from cloudpathlib.anypath import AnyPath


model_name = "sberbank-ai/rugpt3small_based_on_gpt2"


def main(data_dir="data", save_dir="saves", peft=False, deepspeed=True):
    data_dir = AnyPath(data_dir)
    save_dir = AnyPath(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Model parameters: {model.num_parameters():,}")
    if peft:
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
            txt_path, tokenizer, n_ctx: int, max_n_examples: int = None
        ) -> "MurakamiDataset":
            if (pt_path := txt_path.with_suffix(".token_ids.pt")).exists():
                print(f"Loading dataset from {pt_path}")
                ids = torch.load(str(pt_path))
            else:
                with open(txt_path, "r") as f:
                    text = f.read()
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

    test_text_path = data_dir / "murakami_test.txt"
    train_text_path = data_dir / "murakami_train.txt"
    test_set = MurakamiDataset.load(
        test_text_path, tokenizer, model.config.n_ctx, max_n_examples=100
    )
    train_set = MurakamiDataset.load(train_text_path, tokenizer, model.config.n_ctx)

    save_dir = save_dir / f"murakami_rugpt3small{'_peft' if peft else ''}"
    save_dir.mkdir(exist_ok=True, parents=True)
    if last_checkpoint_dir := get_last_checkpoint(str(save_dir)):
        last_checkpoint_dir = Path(last_checkpoint_dir)
        print([t.name for t in last_checkpoint_dir.iterdir()])

    def sample(num_seqs=5, max_length=20):
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
        optimizers=(
            torch.optim.AdamW(model.parameters()),
            torch.optim.lr_scheduler.OneCycleLR,
        ),
        args=TrainingArguments(
            output_dir=str(save_dir),
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else None,
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            save_total_limit=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            ignore_data_skip=True,
            torch_compile=True,
            # https://huggingface.co/docs/accelerate/usage_guides/memory
            auto_find_batch_size=True,
            fp16=True,
        ),
    )
    trainer.train(resume_from_checkpoint=last_checkpoint_dir)


if __name__ == "__main__":
    fire.Fire(main)
