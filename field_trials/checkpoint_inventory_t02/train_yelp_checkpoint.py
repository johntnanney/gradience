#!/usr/bin/env python3
"""Train a small CPU yelp_polarity checkpoint for checkpoint-inventory T02.

Produces:
  experiments/ring2_checkpoint_delta/checkpoints/yelp_s42

Uses cached HuggingFace Arrow files only (no network).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "ring2_checkpoint_delta" / "checkpoints" / "yelp_s42"
DEFAULT_BASE_MODEL = "distilbert-base-uncased"


def _find_yelp_cache_dir() -> Path:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "yelp_polarity" / "plain_text" / "0.0.0"
    if not root.exists():
        raise FileNotFoundError(f"Missing Yelp cache root: {root}")

    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "yelp_polarity-train.arrow").exists() and (candidate / "yelp_polarity-test.arrow").exists():
            return candidate
    raise FileNotFoundError("Could not find Yelp train/test Arrow files in cache")


def _load_yelp_splits() -> tuple[Dataset, Dataset]:
    cache_dir = _find_yelp_cache_dir()
    train_ds = Dataset.from_file(str(cache_dir / "yelp_polarity-train.arrow"))
    test_ds = Dataset.from_file(str(cache_dir / "yelp_polarity-test.arrow"))
    return train_ds, test_ds


class YelpTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer: AutoTokenizer,
        *,
        max_length: int,
    ):
        self.encodings: list[dict[str, torch.Tensor]] = []
        self.labels: list[int] = []

        for row in rows:
            encoded = tokenizer(
                str(row["text"]),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.encodings.append({k: v.squeeze(0) for k, v in encoded.items()})
            self.labels.append(int(row["label"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v.clone() for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _sample_rows(ds: Dataset, n: int, seed: int) -> list[dict]:
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    chosen = indices[: min(n, len(indices))]
    return [ds[i] for i in chosen]


def train_yelp(
    *,
    out_dir: Path,
    base_model: str,
    seed: int,
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_length: int,
    force: bool,
) -> None:
    if out_dir.exists() and (out_dir / "config.json").exists() and not force:
        print(f"[skip] checkpoint already exists: {out_dir}")
        return

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_split, test_split = _load_yelp_splits()
    train_rows = _sample_rows(train_split, train_samples, seed)
    val_rows = _sample_rows(test_split, val_samples, seed + 1)

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        local_files_only=True,
    )
    model.to(torch.device("cpu"))

    train_ds = YelpTorchDataset(train_rows, tokenizer, max_length=max_length)
    val_ds = YelpTorchDataset(val_rows, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    print("Training yelp_s42 checkpoint")
    print(f"  out_dir={out_dir}")
    print(f"  base_model={base_model} train_rows={len(train_rows)} val_rows={len(val_rows)}")
    print(f"  epochs={epochs} batch={batch_size} lr={lr} wd={weight_decay}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += float(loss.item())
        avg_loss = train_loss / max(1, len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(**batch).logits
                pred = logits.argmax(dim=-1)
                correct += int((pred == batch["labels"]).sum().item())
                total += int(batch["labels"].size(0))
        val_acc = correct / total if total else 0.0
        print(f"  epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    elapsed = time.time() - t0
    print(f"[done] saved checkpoint to {out_dir} elapsed={elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train yelp_s42 full checkpoint (CPU)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_yelp(
        out_dir=Path(args.out_dir),
        base_model=args.base_model,
        seed=args.seed,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        force=args.force,
    )


if __name__ == "__main__":
    main()
