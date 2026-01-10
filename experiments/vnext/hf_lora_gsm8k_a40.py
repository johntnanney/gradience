from __future__ import annotations

import argparse, math, os, json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType
from gradience.vnext.integrations import GradienceVNextCallback

def format_gsm8k(ex):
    q = ex["question"].strip()
    a = ex["answer"].strip()
    # Keep it simple: supervised fine-tune with explicit delimiter
    # You can swap this for your rigorous masking later; this is for telemetry plumbing.
    return f"Q: {q}\nA: {a}\n"

def tokenize_and_mask(tokenizer, text, max_len=256):
    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # Minimal masking: everything contributes (you can upgrade to prompt-masking later)
    labels = input_ids.copy()
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

class Collator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
    def __call__(self, batch):
        # Pad input_ids/attention_mask/labels
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tok.pad_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            out["input_ids"].append(torch.tensor(x["input_ids"] + [pad_id]*pad))
            out["attention_mask"].append(torch.tensor(x["attention_mask"] + [0]*pad))
            # label padding should be -100
            out["labels"].append(torch.tensor(x["labels"] + [-100]*pad))
        out = {k: torch.stack(v) for k,v in out.items()}
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/a40_gsm8k")
    ap.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--train-samples", type=int, default=512)
    ap.add_argument("--test-samples", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--telemetry-allow-text", action="store_true", help="Allow logging long strings (e.g., prompts/examples) into telemetry JSONL. Default redacts >256 chars.")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "peft").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(parents=True, exist_ok=True)

    print(f"[run] model={args.model} out={out} steps={args.max_steps} r={args.r} alpha={args.alpha} lr={args.lr}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # works for LLaMA/Mistral/TinyLlama; change for other archs
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("gsm8k", "main")
    train_raw = ds["train"].select(range(min(args.train_samples, len(ds["train"]))))
    test_raw = ds["test"].select(range(min(args.test_samples, len(ds["test"]))))

    train_text = train_raw.map(lambda ex: {"text": format_gsm8k(ex)}, remove_columns=train_raw.column_names)
    test_text  = test_raw.map(lambda ex: {"text": format_gsm8k(ex)}, remove_columns=test_raw.column_names)

    train_tok = train_text.map(lambda ex: tokenize_and_mask(tok, ex["text"], args.max_len), remove_columns=["text"])
    test_tok  = test_text.map(lambda ex: tokenize_and_mask(tok, ex["text"], args.max_len), remove_columns=["text"])

    collator = Collator(tok)

    # vNext telemetry callback
    telemetry_path = str(out / "run.jsonl")
    cb = GradienceVNextCallback(
        telemetry_path=telemetry_path,
        model_name=args.model,
        dataset_name="gsm8k",
        task_profile="hard_reasoning",
        lora_config={"r": args.r, "alpha": args.alpha, "target_modules": lora_cfg.target_modules},
        eval_split="test",   # default eval split label for eval_*
        log_every_n_steps=10,
        telemetry_allow_text=args.telemetry_allow_text,
    )

    targs = TrainingArguments(
        output_dir=str(out / "training"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        weight_decay=0.0,
        max_steps=args.max_steps,
        logging_steps=10,
        eval_strategy="no",  # we run our own evaluate() calls with prefixes
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=collator,
        callbacks=[cb],
    )

    trainer.train()

    # Evaluate train + test explicitly so callback logs split=train and split=test
    train_metrics = trainer.evaluate(eval_dataset=train_tok, metric_key_prefix="train")
    test_metrics  = trainer.evaluate(eval_dataset=test_tok,  metric_key_prefix="test")

    # Save adapter to safetensors
    model.save_pretrained(str(out / "peft"), safe_serialization=True)
    tok.save_pretrained(str(out / "peft"))

    # Save training args for `gradience check --training-dir ...`
    (out / "training" / "training_args.json").write_text(targs.to_json_string(), encoding="utf-8")

    print("\n[done]")
    print(f"Telemetry: {telemetry_path}")
    print(f"PEFT dir:  {out / 'peft'}")
    print(f"Training:  {out / 'training' / 'training_args.json'}")

if __name__ == "__main__":
    main()
