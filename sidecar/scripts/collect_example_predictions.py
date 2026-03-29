#!/usr/bin/env python3
"""Collect per-example predictions and logits for the Output Example Semantics panel.

For each panel case, evaluates source A, source B, and the merged adapter on the
same fixed 500-example slice. Saves predictions, softmax probabilities, and labels.

Output: sidecar/results/example_semantics/predictions/<case_id>.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("collect_preds")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIELD_TRIALS = REPO_ROOT / "field_trials"
OUTPUT_DIR = REPO_ROOT / "sidecar" / "results" / "example_semantics" / "predictions"
MAX_SAMPLES = 500

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "tweet_eval/irony": {
        "hf_name": "tweet_eval", "hf_config": "irony",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 2,
    },
    "tweet_eval/emotion": {
        "hf_name": "tweet_eval", "hf_config": "emotion",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 4,
    },
    "tweet_eval/hate": {
        "hf_name": "tweet_eval", "hf_config": "hate",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 2,
    },
    "ag_news": {
        "hf_name": "ag_news", "hf_config": None,
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 4,
    },
}


# ---------------------------------------------------------------------------
# Panel case definitions
# ---------------------------------------------------------------------------

@dataclass
class PanelCase:
    case_id: str
    case_class: str
    source_a_dir: Path
    source_b_dir: Path
    merged_dir: Path
    base_model: str
    eval_dataset: str


def build_panel() -> list[PanelCase]:
    inv04 = FIELD_TRIALS / "inventory_04_distilbert_irony_cluster" / "adapter_cache"
    inv05 = FIELD_TRIALS / "inventory_05_bert_hate_emotion" / "adapter_cache"
    p2b = FIELD_TRIALS / "phase2b_eval_145914"

    def d(name: str) -> Path:
        return inv04 / name

    def b(name: str) -> Path:
        return inv05 / name

    return [
        PanelCase(
            case_id="SR-01", case_class="safe_retained",
            source_a_dir=d("TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
            source_b_dir=d("TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
            merged_dir=p2b / "irony_JB173_x_neibla" / "merged_adapter",
            base_model="distilbert/distilbert-base-uncased",
            eval_dataset="tweet_eval/irony",
        ),
        PanelCase(
            case_id="SR-02", case_class="safe_retained",
            source_a_dir=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate"),
            source_b_dir=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate"),
            merged_dir=p2b / "bert_hate_TGbase_x_hatexplain" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="tweet_eval/hate",
        ),
        PanelCase(
            case_id="FR-01", case_class="fragile",
            source_a_dir=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion"),
            source_b_dir=b("TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion"),
            merged_dir=p2b / "bert_emotion_TGbase_x_fabriceyhc" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="tweet_eval/emotion",
        ),
        PanelCase(
            case_id="FR-02", case_class="fragile",
            source_a_dir=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion"),
            source_b_dir=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion"),
            merged_dir=p2b / "bert_emotion_TGbase_x_hatexplain_NM" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="tweet_eval/emotion",
        ),
        PanelCase(
            case_id="CT-01", case_class="control",
            source_a_dir=b("TransferGraph__bert-base-uncased-finetuned-lora-ag_news"),
            source_b_dir=b("TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate"),
            merged_dir=p2b / "bert_cross_agnews_x_aviator_hate_CTRL" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="ag_news",
        ),
        PanelCase(
            case_id="NM-01", case_class="near_miss",
            source_a_dir=d("TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
            source_b_dir=d("TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony"),
            merged_dir=p2b / "irony_JB173_x_phailyoor_NM" / "merged_adapter",
            base_model="distilbert/distilbert-base-uncased",
            eval_dataset="tweet_eval/irony",
        ),
        PanelCase(
            case_id="NM-02", case_class="near_miss",
            source_a_dir=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate"),
            source_b_dir=b("TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate"),
            merged_dir=p2b / "bert_hate_TGbase_x_aviator_NM" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="tweet_eval/hate",
        ),
        PanelCase(
            case_id="AN-01", case_class="anchor",
            source_a_dir=b("TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion"),
            source_b_dir=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion"),
            merged_dir=p2b / "bert_emotion_fabriceyhc_x_hatexplain_NM" / "merged_adapter",
            base_model="google-bert/bert-base-uncased",
            eval_dataset="tweet_eval/emotion",
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_adapter(
    adapter_dir: Path,
    base_model_id: str,
    ds_key: str,
    texts: list[str],
    num_labels: int,
) -> tuple[list[int], list[list[float]]]:
    """Evaluate an adapter. Returns (predictions, softmax_probs)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=num_labels
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_preds, all_probs


def process_case(case: PanelCase) -> dict:
    """Process a single panel case: evaluate source A, source B, merged."""
    info = DATASET_REGISTRY[case.eval_dataset]
    ds = load_dataset(
        info["hf_name"], info["hf_config"],
        split=f"{info['split']}[:{MAX_SAMPLES}]"
    )
    texts = list(ds[info["text_col"]])
    labels = list(ds[info["label_col"]])
    num_labels = info["num_labels"]
    n = len(labels)

    log.info("=== %s (%s) ===", case.case_id, case.case_class)

    source_b_available = True  # will be set to False if cross-task mismatch

    # Evaluate source A
    log.info("  Evaluating source A: %s", case.source_a_dir.name[:60])
    t0 = time.time()
    preds_a, probs_a = evaluate_adapter(
        case.source_a_dir, case.base_model, case.eval_dataset, texts, num_labels
    )
    log.info("    done in %.1fs", time.time() - t0)

    # Evaluate source B (may fail for cross-task if num_labels mismatch)
    log.info("  Evaluating source B: %s", case.source_b_dir.name[:60])
    t0 = time.time()
    try:
        preds_b, probs_b = evaluate_adapter(
            case.source_b_dir, case.base_model, case.eval_dataset, texts, num_labels
        )
        log.info("    done in %.1fs", time.time() - t0)
        source_b_available = True
    except RuntimeError as e:
        if "size mismatch" in str(e):
            log.warning("    Source B num_labels mismatch (cross-task). Skipping source B predictions.")
            preds_b = [-1] * n  # sentinel
            probs_b = [[0.0] * num_labels] * n
            source_b_available = False
        else:
            raise

    # Evaluate merged
    log.info("  Evaluating merged: %s", case.merged_dir.name[:60])
    t0 = time.time()
    preds_m, probs_m = evaluate_adapter(
        case.merged_dir, case.base_model, case.eval_dataset, texts, num_labels
    )
    log.info("    done in %.1fs", time.time() - t0)

    # Compute quick summary stats
    acc_a = sum(1 for p, l in zip(preds_a, labels) if p == l) / n
    acc_b = sum(1 for p, l in zip(preds_b, labels) if p == l) / n if source_b_available else None
    acc_m = sum(1 for p, l in zip(preds_m, labels) if p == l) / n

    if source_b_available:
        log.info("  Accuracy: A=%.3f  B=%.3f  Merged=%.3f", acc_a, acc_b, acc_m)
    else:
        log.info("  Accuracy: A=%.3f  B=N/A (cross-task)  Merged=%.3f", acc_a, acc_m)

    result = {
        "case_id": case.case_id,
        "case_class": case.case_class,
        "eval_dataset": case.eval_dataset,
        "base_model": case.base_model,
        "num_labels": num_labels,
        "num_examples": n,
        "source_b_available": source_b_available,
        "accuracy_source_a": round(acc_a, 4),
        "accuracy_source_b": round(acc_b, 4) if acc_b is not None else None,
        "accuracy_merged": round(acc_m, 4),
        "labels": labels,
        "predictions_source_a": preds_a,
        "predictions_source_b": preds_b,
        "predictions_merged": preds_m,
        "probabilities_source_a": [[round(p, 5) for p in row] for row in probs_a],
        "probabilities_source_b": [[round(p, 5) for p in row] for row in probs_b],
        "probabilities_merged": [[round(p, 5) for p in row] for row in probs_m],
    }

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()

    log.info("Collecting per-example predictions for %d panel cases", len(panel))

    for case in panel:
        # Verify paths exist
        for label, path in [("source_a", case.source_a_dir),
                            ("source_b", case.source_b_dir),
                            ("merged", case.merged_dir)]:
            if not path.exists():
                log.error("  %s path missing: %s", label, path)
                sys.exit(1)

        result = process_case(case)

        out_path = OUTPUT_DIR / f"{case.case_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("  Saved → %s", out_path)

    log.info("Done. %d case files written to %s", len(panel), OUTPUT_DIR)


if __name__ == "__main__":
    main()
