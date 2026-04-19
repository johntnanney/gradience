"""N134 Phase 5: Merge evaluation for 69 pairs.

For each pair from pair_sample.json:
  1. Load both adapters' LoRA weights from safetensors
  2. Merge with 0.5/0.5 linear interpolation
  3. Save merged adapter to temp dir
  4. Load base model + merged adapter
  5. Evaluate on both source tasks (task_a and task_b)
  6. Compute degradation: source_acc - merged_acc for each task
  7. Record max_degradation = max(deg_a, deg_b)

Usage:
    python3 scripts/n134/05_merge_eval.py
    python3 scripts/n134/05_merge_eval.py --smoke          # 50 eval samples
    python3 scripts/n134/05_merge_eval.py --pair arc_challenge_s42_vs_hellaswag_s123

Output:
    /workspace/n134/merges/{pair_name}.json     (per-pair)
    /workspace/n134/merges/merge_eval_summary.json  (summary)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
ADAPTER_ROOT = Path("/workspace/n134/adapters")
PAIR_SAMPLE_PATH = Path("/workspace/n134/pair_sample.json")
OUTPUT_DIR = Path("/workspace/n134/merges")
AUDIT_DIR = Path("/workspace/n134/audits")
EVAL_DIR = Path("/workspace/n134/evals")

MAX_SEQ_LEN = 512
EVAL_SAMPLES = 200
SMOKE_EVAL_SAMPLES = 50

# N134 tasks -- all multiple-choice accuracy tasks in [0.50, 0.85] range
TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "openbookqa",
    "commonsenseqa",
    "piqa",
    "siqa",
    "boolq",
]


# ---------------------------------------------------------------------------
# Merge: naive linear average of adapter weights
# ---------------------------------------------------------------------------


def merge_adapters_linear(
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    output_dir: Path,
    alpha: float = 0.5,
) -> None:
    """Create a merged adapter via 0.5/0.5 linear interpolation.

    merged_weight = alpha * weight_a + (1 - alpha) * weight_b
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    weights_a: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_a_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 -- safetensors file, not a dict
            weights_a[k] = f.get_tensor(k)

    weights_b: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_b_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 -- safetensors file, not a dict
            weights_b[k] = f.get_tensor(k)

    merged: dict[str, torch.Tensor] = {}
    for k in weights_a:
        if k in weights_b:
            merged[k] = alpha * weights_a[k] + (1 - alpha) * weights_b[k]
        else:
            merged[k] = weights_a[k]
    for k in weights_b:
        if k not in merged:
            merged[k] = weights_b[k]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_dir / "adapter_model.safetensors"))
    shutil.copy(adapter_a_dir / "adapter_config.json", output_dir / "adapter_config.json")


# ---------------------------------------------------------------------------
# Spectral alignment lookup
# ---------------------------------------------------------------------------


def load_spectral_alignment(adapter_a: str, adapter_b: str) -> float | None:
    """Look up pairwise spectral alignment from audit data."""
    summary_file = AUDIT_DIR / "pair_alignment_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file) as f:
        pair_data = json.load(f)

    # Try both orderings
    for key in [f"{adapter_a}_vs_{adapter_b}", f"{adapter_b}_vs_{adapter_a}"]:
        if key in pair_data:
            return float(pair_data[key].get("mean_alignment", 0.0))
    return None


# ---------------------------------------------------------------------------
# Source score lookup
# ---------------------------------------------------------------------------


def load_source_score(adapter_name: str) -> float | None:
    """Load source eval accuracy for an adapter."""
    # Try dedicated eval file first
    src_file = EVAL_DIR / f"{adapter_name}_source_eval.json"
    if src_file.exists():
        data = json.loads(src_file.read_text())
        return float(data.get("accuracy", 0.0))

    # Try training meta
    meta_file = ADAPTER_ROOT / adapter_name / "training_meta.json"
    if meta_file.exists():
        data = json.loads(meta_file.read_text())
        return float(data.get("eval_accuracy", data.get("accuracy", 0.0)))

    return None


# ---------------------------------------------------------------------------
# Task-specific evaluation data loading
# ---------------------------------------------------------------------------


def load_eval_data(task: str, n_samples: int = EVAL_SAMPLES) -> list[dict]:
    """Load evaluation examples for a task.

    All N134 tasks are multiple-choice. Prompts are formatted WITHOUT the
    answer; the model generates freely and we check if the generation
    matches the expected answer.
    """
    from datasets import load_dataset

    if task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=CACHE_DIR, split="test")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            choice_str = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            prompt = (
                f"Answer the following science question by selecting the correct option.\n\n"
                f"Question: {ex['question']}\n"
                f"Choices:\n{choice_str}\n"
                f"Answer:"
            )
            examples.append({"prompt": prompt, "expected": ex["answerKey"], "task": task})
        return examples

    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            endings = ex["endings"]
            choice_str = "\n".join(f"  {i}. {e}" for i, e in enumerate(endings))
            prompt = f"Select the most plausible continuation.\n\nContext: {ex['ctx']}\nChoices:\n{choice_str}\nAnswer:"
            expected = str(ex["label"])
            examples.append({"prompt": prompt, "expected": expected, "task": task})
        return examples

    elif task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = (
                f"Fill in the blank with the correct option.\n\n"
                f"Sentence: {ex['sentence']}\n"
                f"  1. {ex['option1']}\n"
                f"  2. {ex['option2']}\n"
                f"Answer:"
            )
            examples.append({"prompt": prompt, "expected": ex["answer"], "task": task})
        return examples

    elif task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", cache_dir=CACHE_DIR, split="test")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            choice_str = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            prompt = (
                f"Answer the following science question by selecting the correct option.\n\n"
                f"Question: {ex['question_stem']}\n"
                f"Choices:\n{choice_str}\n"
                f"Answer:"
            )
            examples.append({"prompt": prompt, "expected": ex["answerKey"], "task": task})
        return examples

    elif task == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            choice_str = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            prompt = (
                f"Answer the following question by selecting the correct option.\n\n"
                f"Question: {ex['question']}\n"
                f"Choices:\n{choice_str}\n"
                f"Answer:"
            )
            examples.append({"prompt": prompt, "expected": ex["answerKey"], "task": task})
        return examples

    elif task == "piqa":
        ds = load_dataset("ybisk/piqa", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = (
                f"Which solution is more plausible for the following goal?\n\n"
                f"Goal: {ex['goal']}\n"
                f"  0. {ex['sol1']}\n"
                f"  1. {ex['sol2']}\n"
                f"Answer:"
            )
            expected = str(ex["label"])
            examples.append({"prompt": prompt, "expected": expected, "task": task})
        return examples

    elif task == "siqa":
        ds = load_dataset("lighteval/siqa", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = (
                f"Answer the following social situation question.\n\n"
                f"Context: {ex['context']}\n"
                f"Question: {ex['question']}\n"
                f"  1. {ex['answerA']}\n"
                f"  2. {ex['answerB']}\n"
                f"  3. {ex['answerC']}\n"
                f"Answer:"
            )
            expected = str(ex["label"])
            examples.append({"prompt": prompt, "expected": expected, "task": task})
        return examples

    elif task == "boolq":
        ds = load_dataset("google/boolq", cache_dir=CACHE_DIR, split="validation")
        examples = []
        for ex in ds.select(range(min(n_samples, len(ds)))):
            prompt = (
                f"Answer the following question with True or False.\n\n"
                f"Passage: {ex['passage'][:800]}\n"
                f"Question: {ex['question']}\n"
                f"Answer:"
            )
            expected = "True" if ex["answer"] else "False"
            examples.append({"prompt": prompt, "expected": expected, "task": task})
        return examples

    return []


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_model_on_task(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    task: str,
    n_samples: int = EVAL_SAMPLES,
) -> dict:
    """Evaluate a model on a task. Returns accuracy, correct, total."""
    examples = load_eval_data(task, n_samples)
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = 0
    total = 0

    for ex in examples:
        inputs = tokenizer(
            ex["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        # Check if expected answer appears in generation
        expected = ex["expected"]
        gen_lower = generated.lower()[:50]
        exp_lower = expected.lower()

        if task == "boolq":
            correct += int(exp_lower in gen_lower)
        elif task in ("hellaswag", "piqa", "siqa", "winogrande"):
            # Numeric label tasks: check first non-whitespace character
            gen_stripped = generated.strip()
            correct += int(gen_stripped.startswith(expected))
        else:
            # Letter label tasks (arc, openbookqa, commonsenseqa)
            correct += int(exp_lower in gen_lower)

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": round(accuracy, 6), "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------


def evaluate_pair(
    pair_info: dict,
    model_name: str,
    tokenizer: AutoTokenizer,
    n_eval: int,
) -> dict:
    """Evaluate a single merged pair. Returns the result dict."""
    adapter_a = pair_info["adapter_a"]
    adapter_b = pair_info["adapter_b"]
    task_a = pair_info["task_a"]
    task_b = pair_info["task_b"]
    category = pair_info["category"]
    pair_name = f"{adapter_a}_vs_{adapter_b}"

    adapter_a_dir = ADAPTER_ROOT / adapter_a
    adapter_b_dir = ADAPTER_ROOT / adapter_b
    merge_dir = OUTPUT_DIR / f"merged_{pair_name}"

    # Merge adapters
    merge_adapters_linear(adapter_a_dir, adapter_b_dir, merge_dir)

    # Load base model + merged adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(merge_dir))
    model.eval()

    # Evaluate on both source tasks
    tasks_to_eval = sorted(set([task_a, task_b]))
    eval_results: dict[str, dict] = {}
    for task in tasks_to_eval:
        print(f"    Evaluating on {task}...")
        eval_results[task] = evaluate_model_on_task(model, tokenizer, task, n_samples=n_eval)
        print(
            f"    {task}: {eval_results[task]['accuracy']:.4f} "
            f"({eval_results[task]['correct']}/{eval_results[task]['total']})"
        )

    # Load source scores
    source_scores: dict[str, float | None] = {}
    source_scores[adapter_a] = load_source_score(adapter_a)
    source_scores[adapter_b] = load_source_score(adapter_b)

    # Spectral alignment
    spectral_alignment = load_spectral_alignment(adapter_a, adapter_b)

    # Compute degradation per task
    degradations: dict[str, float] = {}
    for task in tasks_to_eval:
        merged_acc = eval_results[task]["accuracy"]
        # Find the source adapter for this task
        best_source = 0.0
        for aname in [adapter_a, adapter_b]:
            atask = task_a if aname == adapter_a else task_b
            if atask == task and source_scores.get(aname) is not None:
                best_source = max(best_source, source_scores[aname])
        degradations[task] = round(best_source - merged_acc, 6)

    deg_values = list(degradations.values())
    max_deg = max(deg_values) if deg_values else 0.0
    min_deg = min(deg_values) if deg_values else 0.0
    mean_deg = sum(deg_values) / len(deg_values) if deg_values else 0.0

    result: dict = {
        "pair": pair_name,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "task_a": task_a,
        "task_b": task_b,
        "category": category,
    }

    if "family_pair" in pair_info:
        result["family_pair"] = pair_info["family_pair"]

    result["spectral_alignment"] = spectral_alignment
    result["merge_eval"] = eval_results

    # Source scores as floats (None -> null in JSON)
    result["source_scores"] = {k: v for k, v in source_scores.items()}

    for task in tasks_to_eval:
        result[f"degradation_{task}"] = degradations[task]

    result["max_degradation"] = round(max_deg, 6)
    result["min_degradation"] = round(min_deg, 6)
    result["mean_degradation"] = round(mean_deg, 6)
    result["asymmetry"] = round(max_deg - min_deg, 6)

    # Cleanup model
    del model
    torch.cuda.empty_cache()

    # Remove merged adapter files
    if merge_dir.exists():
        shutil.rmtree(merge_dir)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="N134 merge evaluation for 69 pairs")
    parser.add_argument(
        "--smoke", action="store_true", help=f"Use {SMOKE_EVAL_SAMPLES} eval samples instead of {EVAL_SAMPLES}"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Evaluate a single pair by name (e.g. arc_challenge_s42_vs_hellaswag_s123)",
    )
    parser.add_argument("--n-eval", type=int, default=None, help="Override eval sample count")
    args = parser.parse_args()

    n_eval = args.n_eval or (SMOKE_EVAL_SAMPLES if args.smoke else EVAL_SAMPLES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pair sample
    if not PAIR_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Pair sample not found: {PAIR_SAMPLE_PATH}\nRun 04_sample_pairs.py first.")

    with open(PAIR_SAMPLE_PATH) as f:
        pair_sample = json.load(f)

    all_pairs: list[dict] = pair_sample["all_pairs"]

    # Filter to single pair if requested
    if args.pair:
        matched = [p for p in all_pairs if f"{p['adapter_a']}_vs_{p['adapter_b']}" == args.pair]
        if not matched:
            # Try reversed order
            matched = [p for p in all_pairs if f"{p['adapter_b']}_vs_{p['adapter_a']}" == args.pair]
        if not matched:
            raise ValueError(f"Pair not found: {args.pair}")
        all_pairs = matched

    print("=" * 60)
    print("  N134 Phase 5: Merge Evaluation")
    print("=" * 60)
    print(f"  Pairs to evaluate: {len(all_pairs)}")
    print(f"  Eval samples/task: {n_eval}")
    if args.smoke:
        print("  Mode:              SMOKE")
    print()

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[dict] = []
    skipped = 0
    t0 = time.time()

    for i, pair_info in enumerate(all_pairs):
        adapter_a = pair_info["adapter_a"]
        adapter_b = pair_info["adapter_b"]
        pair_name = f"{adapter_a}_vs_{adapter_b}"
        result_file = OUTPUT_DIR / f"{pair_name}.json"

        # Idempotence: skip if result already exists
        if result_file.exists():
            print(f"  [{i + 1}/{len(all_pairs)}] SKIP {pair_name} -- already evaluated")
            results.append(json.loads(result_file.read_text()))
            skipped += 1
            continue

        print(f"\n  [{i + 1}/{len(all_pairs)}] Evaluating {pair_name}...")
        t_pair = time.time()

        result = evaluate_pair(pair_info, MODEL_NAME, tokenizer, n_eval)

        # Save per-pair result
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        results.append(result)
        elapsed_pair = time.time() - t_pair
        print(f"    max_degradation: {result['max_degradation']:.4f}  ({elapsed_pair:.0f}s)")

    elapsed = time.time() - t0

    # Save summary
    summary = {
        "n_pairs": len(results),
        "n_evaluated": len(results) - skipped,
        "n_skipped": skipped,
        "eval_samples_per_task": n_eval,
        "smoke": args.smoke,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }

    summary_path = OUTPUT_DIR / "merge_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Completed: {len(results) - skipped} evaluated, {skipped} skipped")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
