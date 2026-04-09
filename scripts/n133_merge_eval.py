"""N133 Phase 3: Merge evaluation for priority pairs.

Evaluates naive linear (0.5/0.5) merges for:
  - All 6 same-task pairs
  - Top 6 cross-task pairs by compatibility (from audit)
  - Bottom 6 cross-task pairs by compatibility

Usage:
    python3 n133_merge_eval.py
    python3 n133_merge_eval.py --all     # evaluate all 66 pairs
    python3 n133_merge_eval.py --smoke   # quick test

Output: /workspace/n133/merges/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
ADAPTER_ROOT = Path("/workspace/n133/adapters")
AUDIT_DIR = Path("/workspace/n133/audits")
OUTPUT_DIR = Path("/workspace/n133/merges")
EVAL_DIR = Path("/workspace/n133/evals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 512
EVAL_SAMPLES = 200  # per task for merge eval


# ---------------------------------------------------------------------------
# Merge: naive linear average of adapter weights
# ---------------------------------------------------------------------------

def merge_adapters_linear(adapter_a_dir: Path, adapter_b_dir: Path,
                          output_dir: Path, alpha: float = 0.5):
    """Create a merged adapter via linear interpolation."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Load both sets of weights
    weights_a = {}
    with safe_open(str(adapter_a_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f:
            weights_a[k] = f.get_tensor(k)

    weights_b = {}
    with safe_open(str(adapter_b_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f:
            weights_b[k] = f.get_tensor(k)

    # Linear interpolation
    merged = {}
    for k in weights_a:
        if k in weights_b:
            merged[k] = alpha * weights_a[k] + (1 - alpha) * weights_b[k]
        else:
            merged[k] = weights_a[k]
    for k in weights_b:
        if k not in merged:
            merged[k] = weights_b[k]

    # Save merged adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_dir / "adapter_model.safetensors"))

    # Copy adapter config from adapter_a
    import shutil
    shutil.copy(adapter_a_dir / "adapter_config.json", output_dir / "adapter_config.json")


# ---------------------------------------------------------------------------
# Task-specific evaluation
# ---------------------------------------------------------------------------

def load_eval_data(task: str, n_samples: int = EVAL_SAMPLES) -> list[dict]:
    """Load evaluation examples for a task."""
    from datasets import load_dataset

    if task == "sst2":
        ds = load_dataset("glue", "sst2", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = f"Classify the sentiment of this sentence as positive or negative.\n\nSentence: {ex['sentence']}\nSentiment:"
            label = "positive" if ex["label"] == 1 else "negative"
            examples.append({"prompt": prompt, "expected": label, "task": "sst2"})
        return examples

    elif task == "mnli":
        ds = load_dataset("glue", "mnli", cache_dir=CACHE_DIR, split="validation_matched")
        labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = (f"Determine the relationship between the premise and hypothesis.\n\n"
                      f"Premise: {ex['premise']}\n"
                      f"Hypothesis: {ex['hypothesis']}\n"
                      f"Relationship:")
            examples.append({"prompt": prompt, "expected": labels[ex["label"]], "task": "mnli"})
        return examples

    elif task == "squad":
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            answers = ex["answers"]
            if not answers["text"]:
                continue
            prompt = (f"Answer the question based on the context.\n\n"
                      f"Context: {ex['context'][:800]}\n"
                      f"Question: {ex['question']}\n"
                      f"Answer:")
            examples.append({"prompt": prompt, "expected": answers["text"][0], "task": "squad"})
            if len(examples) >= n_samples:
                break
        return examples

    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", cache_dir=CACHE_DIR, split="test")
        examples = []
        import re
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = f"Solve the following math problem step by step.\n\nProblem: {ex['question']}\nSolution:"
            # Extract final answer
            nums = re.findall(r"####\s*([\d,]+)", ex["answer"])
            expected = nums[-1].replace(",", "") if nums else ""
            examples.append({"prompt": prompt, "expected": expected, "task": "gsm8k"})
        return examples

    elif task == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", cache_dir=CACHE_DIR, split="train")
        # Use last 500 as eval
        n = len(ds)
        examples = []
        for ex in ds.select(range(max(0, n - 500), min(n, n - 500 + n_samples))):
            inp = ex.get("input", "")
            instruction = ex["instruction"]
            if inp and inp.strip():
                instruction = f"{instruction}\n\nInput: {inp}"
            prompt = f"Write code to solve the following task.\n\nTask: {instruction}\nCode:\n"
            examples.append({"prompt": prompt, "expected": ex["output"], "task": "code"})
        return examples

    elif task == "summarization":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir=CACHE_DIR, split="test")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = f"Summarize the following article.\n\nArticle: {ex['article'][:1500]}\n\nSummary:"
            examples.append({"prompt": prompt, "expected": ex["highlights"], "task": "summarization"})
        return examples

    return []


def evaluate_model_on_task(model, tokenizer, task: str, n_samples: int = EVAL_SAMPLES) -> dict:
    """Evaluate a model on a task, return accuracy/score."""
    examples = load_eval_data(task, n_samples)
    if not examples:
        return {"accuracy": 0, "total": 0, "correct": 0}

    correct = 0
    total = 0
    import re

    for ex in examples:
        inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            max_new = 20 if task in ("sst2", "mnli") else 100
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

        if task in ("sst2", "mnli"):
            correct += int(ex["expected"].lower() in generated.lower()[:50])
        elif task == "gsm8k":
            gen_nums = re.findall(r"[\d]+", generated.replace(",", ""))
            correct += int(bool(gen_nums) and gen_nums[-1] == ex["expected"])
        elif task == "squad":
            # Token F1
            expected_tokens = set(ex["expected"].lower().split())
            generated_tokens = set(generated.lower().split()[:len(expected_tokens) * 2])
            if expected_tokens and generated_tokens:
                precision = len(expected_tokens & generated_tokens) / len(generated_tokens)
                recall = len(expected_tokens & generated_tokens) / len(expected_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                correct += int(f1 > 0.5)
            else:
                correct += 0
        else:
            correct += int(len(generated) > 10)

        total += 1

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Main merge evaluation loop
# ---------------------------------------------------------------------------

def select_priority_pairs() -> list[tuple[str, str]]:
    """Select 18 priority pairs: 6 same-task + top 6 + bottom 6 cross-task."""
    audit_file = AUDIT_DIR / "pair_alignment_summary.json"
    if not audit_file.exists():
        raise FileNotFoundError(f"Run spectral audit first: {audit_file}")

    with open(audit_file) as f:
        pair_data = json.load(f)

    same_task = []
    cross_task = []
    for pair_key, info in pair_data.items():
        item = (info["adapter_a"], info["adapter_b"], info["mean_alignment"], pair_key)
        if info["is_same_task"]:
            same_task.append(item)
        else:
            cross_task.append(item)

    # Sort cross-task by alignment (high = compatible, low = incompatible)
    cross_task.sort(key=lambda x: x[2], reverse=True)

    # Select: all same-task + top 6 cross-task + bottom 6 cross-task
    selected = []
    for a, b, align, _key in same_task:
        selected.append((a, b, "same_task", align))
    for a, b, align, _key in cross_task[:6]:
        selected.append((a, b, "cross_top", align))
    for a, b, align, _key in cross_task[-6:]:
        selected.append((a, b, "cross_bottom", align))

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Evaluate all 66 pairs")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 pairs)")
    parser.add_argument("--n-eval", type=int, default=EVAL_SAMPLES, help="Eval samples per task")
    args = parser.parse_args()

    print("=" * 60)
    print("  N133 Phase 3: Merge Evaluation")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select pairs
    if args.smoke:
        pairs = select_priority_pairs()[:2]
    elif args.all:
        # All pairs from audit
        with open(AUDIT_DIR / "pair_alignment_summary.json") as f:
            pair_data = json.load(f)
        pairs = [(v["adapter_a"], v["adapter_b"],
                   "same_task" if v["is_same_task"] else "cross_task",
                   v["mean_alignment"])
                  for v in pair_data.values()]
    else:
        pairs = select_priority_pairs()

    print(f"\n  Evaluating {len(pairs)} pairs ({args.n_eval} samples per task)")

    results = []
    t0 = time.time()

    for i, (adapter_a, adapter_b, category, spectral_align) in enumerate(pairs):
        pair_name = f"{adapter_a}_vs_{adapter_b}"
        result_file = OUTPUT_DIR / f"{pair_name}.json"

        if result_file.exists():
            print(f"\n  [{i+1}/{len(pairs)}] SKIP {pair_name} — already evaluated")
            results.append(json.loads(result_file.read_text()))
            continue

        print(f"\n  [{i+1}/{len(pairs)}] Merging {pair_name}...")
        adapter_a_dir = ADAPTER_ROOT / adapter_a
        adapter_b_dir = ADAPTER_ROOT / adapter_b
        merge_dir = OUTPUT_DIR / f"merged_{pair_name}"

        # Create merge
        merge_adapters_linear(adapter_a_dir, adapter_b_dir, merge_dir)

        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(merge_dir))
        model.eval()

        # Evaluate on both source tasks
        task_a = adapter_a.split("_")[0]
        task_b = adapter_b.split("_")[0]
        tasks_to_eval = sorted(set([task_a, task_b]))

        eval_results = {}
        for task in tasks_to_eval:
            print(f"    Evaluating on {task}...")
            eval_results[task] = evaluate_model_on_task(
                model, tokenizer, task, n_samples=args.n_eval
            )
            print(f"    {task}: {eval_results[task]['accuracy']:.3f} "
                  f"({eval_results[task]['correct']}/{eval_results[task]['total']})")

        # Load source scores for comparison
        source_scores = {}
        for adapter_name in [adapter_a, adapter_b]:
            src_file = EVAL_DIR / f"{adapter_name}_source_eval.json"
            if src_file.exists():
                source_scores[adapter_name] = json.loads(src_file.read_text())

        result = {
            "pair": pair_name,
            "adapter_a": adapter_a,
            "adapter_b": adapter_b,
            "task_a": task_a,
            "task_b": task_b,
            "category": category,
            "spectral_alignment": spectral_align,
            "merge_eval": eval_results,
            "source_scores": source_scores,
        }

        # Compute degradation
        for task in tasks_to_eval:
            merged_acc = eval_results[task]["accuracy"]
            best_source = 0
            for adapter_name in [adapter_a, adapter_b]:
                if adapter_name in source_scores:
                    src = source_scores[adapter_name]
                    if src.get("task") == task:
                        best_source = max(best_source, src.get("accuracy", 0))
            result[f"degradation_{task}"] = best_source - merged_acc

        results.append(result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        # Remove merged adapter files to save space
        import shutil
        if merge_dir.exists():
            shutil.rmtree(merge_dir)

    elapsed = time.time() - t0
    print(f"\n  Total eval time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save summary
    summary = {
        "n_pairs": len(results),
        "elapsed_s": elapsed,
        "results": results,
    }
    with open(OUTPUT_DIR / "merge_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
