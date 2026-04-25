"""GPU-side inference driver. SPEC_GPU_v0_1 §5–7.

Reads the locked manifests and prompts, runs each pending condition through
LL or G&P scoring, and writes contract-conformant outputs to
runs/raw/{condition_id}/. Stays an inference-only black box: the CPU
pipeline does not depend on this script's implementation choices, only on
the JSON schemas it emits (SPEC_CPU_v0_2 §5.2–5.4).

Usage:

    # Pre-flight only (no GPU work; verifies environment):
    python scripts/gpu_inference.py --preflight-only \
        --config configs/study_config.yaml

    # Primary run (greedy; resumable):
    python scripts/gpu_inference.py \
        --config configs/study_config.yaml \
        --conditions manifests/conditions_primary.csv \
        --conditions-secondary manifests/conditions_gsm8k.csv \
        --fewshot manifests/fewshot_manifest.csv \
        --out-dir runs/raw/

Resume: if runs/raw/{condition_id}/run_metadata.json exists, skip.
Atomicity: each condition writes to runs/raw/.tmp/{condition_id}/, validates,
then atomic-renames to runs/raw/{condition_id}/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_ROOT))

from gradience_study.config import (  # noqa: E402
    BenchmarkSpec,
    MergedConfig,
    ModelSpec,
    PromptSpec,
    ScoringRuleSpec,
    load_config,
)


# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------

PARSE_STATUSES = frozenset({
    "parsed", "unparseable", "empty",
    "multiple_answers", "invalid_choice", "runtime_error",
})

# Per-benchmark item-id field used to namespace IDs at fewshot draw time.
# Mirrors run_phase3_fewshot.NAMESPACE_FIELDS so the GPU side can match
# the CPU side's split-prefixed item IDs in fewshot_manifest.csv.
NAMESPACE_FIELDS: dict[str, str] = {
    "Rowan/hellaswag": "ind",
    "winogrande": "sentence",
    "truthful_qa": "question",
    "gsm8k": "question",
    "cais/mmlu": "question",
    "allenai/ai2_arc": "id",
}

# When emitting metadata.
INFERENCE_BACKEND = "custom_transformers_v0.1"


# ---------------------------------------------------------------------------
# Per-benchmark adapters: extract question, choices, gold, native_id, subject
# ---------------------------------------------------------------------------


def get_question(item: dict, benchmark_id: str) -> str:
    if benchmark_id == "hellaswag":
        # lm-eval style: ctx_a + " " + ctx_b (ctx alone misses the second clause).
        ctx_a = item.get("ctx_a") or ""
        ctx_b = item.get("ctx_b") or ""
        if ctx_a or ctx_b:
            return f"{ctx_a} {ctx_b}".strip()
        return item.get("ctx", "")
    if benchmark_id == "winogrande":
        return item["sentence"]
    return item["question"]


def get_choices(item: dict, benchmark_id: str) -> list[str]:
    if benchmark_id == "arc_challenge":
        return list(item["choices"]["text"])
    if benchmark_id == "hellaswag":
        return list(item["endings"])
    if benchmark_id == "truthfulqa_mc":
        return list(item["mc1_targets"]["choices"])
    if benchmark_id == "mmlu_panel":
        return list(item["choices"])
    if benchmark_id == "winogrande":
        return [item["option1"], item["option2"]]
    if benchmark_id == "gsm8k":
        raise ValueError("gsm8k is open_generation; no choices")
    raise ValueError(f"Unknown benchmark_id: {benchmark_id}")


def get_gold_letter(item: dict, benchmark_id: str) -> str:
    """Constrained-choice benchmarks only. Returns 'A', 'B', ..."""
    if benchmark_id == "arc_challenge":
        labels = list(item["choices"]["label"])
        idx = labels.index(item["answerKey"])
        return chr(ord("A") + idx)
    if benchmark_id == "hellaswag":
        return chr(ord("A") + int(item["label"]))
    if benchmark_id == "truthfulqa_mc":
        labels = list(item["mc1_targets"]["labels"])
        return chr(ord("A") + labels.index(1))
    if benchmark_id == "mmlu_panel":
        return chr(ord("A") + int(item["answer"]))
    if benchmark_id == "winogrande":
        # answer is "1" or "2"
        return "AB"[int(item["answer"]) - 1]
    raise ValueError(
        f"get_gold_letter not defined for {benchmark_id} (open_generation?)"
    )


def get_gold_text(item: dict, benchmark_id: str) -> str:
    """Open-generation benchmarks. Returns the canonical answer string."""
    if benchmark_id == "gsm8k":
        # GSM8K answers end with `#### <number>`; the number is the gold.
        m = re.search(r"####\s*([^\s]+)", item["answer"])
        if m:
            return m.group(1).replace(",", "").strip()
        return item["answer"].strip()
    raise ValueError(
        f"get_gold_text not defined for {benchmark_id} (constrained_choice?)"
    )


def get_native_id(item: dict, benchmark: BenchmarkSpec) -> str:
    """Native item ID as it appears in fewshot_manifest.csv (split-prefixed)."""
    return str(item[benchmark.item_id_field])


def get_subject(item: dict, benchmark_id: str) -> str | None:
    if benchmark_id == "mmlu_panel":
        return item.get("subject")
    return None


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_choices_block(choices: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))


def render_fewshot_example(item: dict, benchmark_id: str) -> str:
    """One block, ending with a trailing newline-newline separator."""
    if benchmark_id == "gsm8k":
        return f"Question: {item['question']}\nAnswer: {item['answer']}\n\n"
    q = get_question(item, benchmark_id)
    choices = get_choices(item, benchmark_id)
    letter = get_gold_letter(item, benchmark_id)
    return (
        f"Question: {q}\n"
        f"{render_choices_block(choices)}\n"
        f"Answer: {letter}\n\n"
    )


def render_prompt(
    template_text: str,
    item: dict,
    fewshot_items: list[dict],
    benchmark: BenchmarkSpec,
    prompt_spec: PromptSpec,
) -> str:
    fewshot_block = "".join(
        render_fewshot_example(fs, benchmark.benchmark_id) for fs in fewshot_items
    )
    rendered = template_text.replace("{{fewshot_examples}}", fewshot_block)
    if "{{question}}" in rendered:
        rendered = rendered.replace("{{question}}", get_question(item, benchmark.benchmark_id))
    if "{{choices}}" in rendered:
        if benchmark.task_type == "open_generation":
            rendered = rendered.replace("{{choices}}", "")
        else:
            choices = get_choices(item, benchmark.benchmark_id)
            rendered = rendered.replace("{{choices}}", render_choices_block(choices))
    # answer_instruction is benchmark-/prompt-specific imperative text. The
    # locked prompts I've inspected leave this as a literal placeholder in
    # the template only when something benchmark-specific belongs there;
    # for our locked set the substitution is empty by default. If a future
    # prompt uses a non-empty answer_instruction it must be carried via
    # PromptSpec.notes (current PromptSpec has no dedicated field).
    if "{{answer_instruction}}" in rendered:
        rendered = rendered.replace("{{answer_instruction}}", "")

    # Defensive: any unfilled placeholder is a bug — fail loudly rather than
    # send a literal `{{...}}` to the model.
    leftover = re.search(r"\{\{[^}]+\}\}", rendered)
    if leftover:
        raise RuntimeError(
            f"Unfilled placeholder in rendered prompt: {leftover.group(0)} "
            f"(benchmark={benchmark.benchmark_id}, prompt={prompt_spec.prompt_id})"
        )
    return rendered


# ---------------------------------------------------------------------------
# Parsing for G&P scoring
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-Z])\b")
_GSM_STRICT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")
_GSM_PERMISSIVE_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def parse_answer(
    raw: str,
    parse_strategy: str,
    n_choices: int | None,
) -> tuple[str, str | None]:
    """Parse a generation per the locked strategy. Returns (status, parsed)."""
    if raw is None or not raw.strip():
        return "empty", None

    if parse_strategy == "benchmark_specific_regex":
        letters = _LETTER_RE.findall(raw)
        if not letters:
            return "unparseable", None
        first = letters[0]
        if n_choices is not None and (ord(first) - ord("A")) >= n_choices:
            return "invalid_choice", first
        # If multiple distinct letters appear early, flag — except a model
        # often emits 'A) ...' style explanations after the answer; we look
        # only at the first 5 letter tokens for "multiple distinct" detection.
        early = letters[:5]
        if len(set(early)) > 1:
            return "multiple_answers", first
        return "parsed", first

    if parse_strategy == "exact_match_final_number":
        m = _GSM_STRICT_RE.search(raw.rstrip())
        if not m:
            return "unparseable", None
        cleaned = m.group(1).replace(",", "")
        return "parsed", cleaned

    if parse_strategy == "permissive_number_regex":
        # Search the last 5 lines for any number; take the last match.
        tail = "\n".join(raw.rstrip().split("\n")[-5:])
        nums = _GSM_PERMISSIVE_RE.findall(tail)
        if not nums:
            return "unparseable", None
        last = nums[-1].replace(",", "").rstrip(".")
        if not last or last in {"-", "+"}:
            return "unparseable", None
        return "parsed", last

    raise ValueError(f"Unknown parse_strategy: {parse_strategy!r}")


# ---------------------------------------------------------------------------
# Scoring (LL and G&P)
# ---------------------------------------------------------------------------


def score_ll_norm(
    model: Any, tokenizer: Any, prompt: str, n_choices: int,
) -> dict[str, Any]:
    """Length-normalized log-likelihood over " A", " B", ... continuations."""
    import torch

    candidates = [f" {chr(ord('A') + i)}" for i in range(n_choices)]
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_len = int(prompt_ids.shape[1])

    scores: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    for letter_idx, candidate in enumerate(candidates):
        letter = chr(ord("A") + letter_idx)
        cand_ids_list = tokenizer.encode(candidate, add_special_tokens=False)
        if not cand_ids_list:
            scores[letter] = float("-inf")
            token_counts[letter] = 1  # schema requires minimum 1
            continue
        cand_ids_t = torch.tensor([cand_ids_list], dtype=torch.long, device=model.device)
        full_ids = torch.cat([prompt_ids, cand_ids_t], dim=1)
        with torch.no_grad():
            logits = model(full_ids).logits  # [1, seq_len, vocab]
        # Position prompt_len-1 predicts the first continuation token.
        slice_logits = logits[0, prompt_len - 1 : prompt_len - 1 + len(cand_ids_list)]
        log_probs = torch.log_softmax(slice_logits, dim=-1)
        token_lps = [
            float(log_probs[t, cand_ids_list[t]].item())
            for t in range(len(cand_ids_list))
        ]
        sum_lp = sum(token_lps)
        scores[letter] = sum_lp / len(cand_ids_list)
        token_counts[letter] = len(cand_ids_list)

    selected = max(scores, key=lambda k: scores[k])
    return {
        "choices": [chr(ord("A") + i) for i in range(n_choices)],
        "choice_scores": scores,
        "choice_token_counts": token_counts,
        "normalization": "length_normalized",
        "selected_answer": selected,
    }


def score_generate_parse(
    model: Any,
    tokenizer: Any,
    prompt: str,
    parse_strategy: str,
    n_choices: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy generate, then parse per strategy."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    prompt_len = int(inputs.input_ids.shape[1])
    gen_ids = outputs[0, prompt_len:]
    raw_generation = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parse_status, parsed_answer = parse_answer(raw_generation, parse_strategy, n_choices)
    return {
        "raw_generation": raw_generation,
        "parsed_answer": parsed_answer,
        "parse_status": parse_status,
        "generation_length_tokens": int(gen_ids.shape[0]),
    }


# ---------------------------------------------------------------------------
# Manifest + few-shot loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionRow:
    condition_id: str
    tier: str
    benchmark_id: str
    subject_id: str | None
    model_id: str
    prompt_id: str
    seed_id: str | None
    fewshot_k: int
    scoring_rule_id: str
    task_type: str
    expected_num_items: int
    condition_status: str
    config_hash: str


def load_conditions(path: Path) -> list[ConditionRow]:
    out: list[ConditionRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(ConditionRow(
                condition_id=row["condition_id"],
                tier=row["tier"],
                benchmark_id=row["benchmark_id"],
                subject_id=row["subject_id"] or None,
                model_id=row["model_id"],
                prompt_id=row["prompt_id"],
                seed_id=row["seed_id"] or None,
                fewshot_k=int(row["fewshot_k"]),
                scoring_rule_id=row["scoring_rule_id"],
                task_type=row["task_type"],
                expected_num_items=int(row["expected_num_items"]),
                condition_status=row["condition_status"],
                config_hash=row["config_hash"],
            ))
    return out


def _normalize_seed(seed_id: str | None) -> str | None:
    """Map condition-manifest seed_id ('s42', 's123', '0shot', '') and
    fewshot-manifest seed_id ('42', '123', '') onto a single canonical form.

    Returns the bare integer string ('42') or None for 0-shot / blank.
    """
    if seed_id is None or seed_id in {"", "0shot"}:
        return None
    if seed_id.startswith("s") and seed_id[1:].lstrip("-").isdigit():
        return seed_id[1:]
    return seed_id


def load_fewshot_manifest(path: Path) -> dict[tuple[str, str | None, str | None], list[str]]:
    """Returns {(benchmark_id, subject_id_or_None, seed_id_or_None): [native_id, ...]}.

    Seeds are canonicalized via _normalize_seed so condition-manifest 's42'
    and fewshot-manifest '42' both resolve to '42'.

    The native IDs are split-prefixed (e.g. "train:Mercury_7230195"); the
    GPU side strips the prefix when looking up an item in the loaded split.
    """
    by_key: dict[tuple[str, str | None, str | None], list[tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = row["benchmark_id"]
            subj = row["subject_id"] or None
            seed = _normalize_seed(row["seed_id"] or None)
            rank = int(row["example_rank"])
            iid = row["example_item_id"]
            by_key.setdefault((bench, subj, seed), []).append((rank, iid))
    # Sort each list by rank; return just the IDs.
    out: dict[tuple[str, str | None, str | None], list[str]] = {}
    for k, lst in by_key.items():
        lst.sort()
        out[k] = [iid for _, iid in lst]
    return out


# ---------------------------------------------------------------------------
# Dataset loading (mirrors run_phase3_fewshot.real_hf_loader)
# ---------------------------------------------------------------------------


def hf_load_split(benchmark: BenchmarkSpec, split: str) -> list[dict]:
    """Load a benchmark split, pinned to the locked dataset_version_hash.

    Mirrors the loader-side adjustments documented in
    IMPLEMENTATION_DEVIATIONS (D-... entries for run_phase3_fewshot):
      - cais/mmlu: hf_config=None becomes "all"; subject filter happens
        downstream by row.
      - All benchmarks: native item-ID field is namespaced with "{split}:"
        prefix so IDs match fewshot_manifest entries.
    """
    from datasets import load_dataset
    cfg = benchmark.hf_config
    if benchmark.hf_dataset == "cais/mmlu" and cfg is None:
        cfg = "all"
    ds = load_dataset(
        benchmark.hf_dataset,
        cfg,
        split=split,
        revision=benchmark.dataset_version_hash,
    )
    rows = [dict(item) for item in ds]
    field = NAMESPACE_FIELDS.get(benchmark.hf_dataset)
    if field:
        for row in rows:
            v = row.get(field)
            if v is not None:
                row[field] = f"{split}:{v}"
    return rows


def filter_to_subject(items: list[dict], subject_id: str | None) -> list[dict]:
    if subject_id is None:
        return items
    return [it for it in items if it.get("subject") == subject_id]


# ---------------------------------------------------------------------------
# Per-condition execution
# ---------------------------------------------------------------------------


@dataclass
class ConditionContext:
    """Per-condition resources resolved before scoring loop."""
    benchmark: BenchmarkSpec
    prompt_spec: PromptSpec
    scoring_rule: ScoringRuleSpec
    template_text: str
    template_hash: str
    eval_items: list[dict]
    fewshot_items: list[dict]
    fewshot_k: int


def _load_template(prompt_spec: PromptSpec) -> tuple[str, str]:
    path = PAPER_ROOT / prompt_spec.prompt_path
    text = path.read_text(encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if h != prompt_spec.content_hash:
        raise RuntimeError(
            f"Prompt content hash mismatch for {prompt_spec.prompt_id}: "
            f"declared {prompt_spec.content_hash}, actual {h}"
        )
    return text, h


def _resolve_fewshot_items(
    condition: ConditionRow,
    benchmark: BenchmarkSpec,
    fewshot_lookup: dict[tuple[str, str | None, str | None], list[str]],
    fewshot_pool_cache: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    if condition.fewshot_k == 0:
        return []
    src_split = benchmark.fewshot_source_split
    assert src_split is not None, "fewshot_k > 0 but fewshot_source_split is None"
    cache_key = (benchmark.benchmark_id, src_split)
    if cache_key not in fewshot_pool_cache:
        fewshot_pool_cache[cache_key] = hf_load_split(benchmark, src_split)
    pool = fewshot_pool_cache[cache_key]
    by_id = {get_native_id(it, benchmark): it for it in pool}

    seed_for_lookup = _normalize_seed(condition.seed_id)
    key = (condition.benchmark_id, condition.subject_id, seed_for_lookup)
    if key not in fewshot_lookup:
        # MMLU "null" subject case won't apply here because MMLU has subjects;
        # other benchmarks with no subject use subject_id=None which matches.
        raise RuntimeError(f"Fewshot manifest missing entry for {key}")
    ids = fewshot_lookup[key]
    if len(ids) != condition.fewshot_k:
        raise RuntimeError(
            f"Fewshot manifest length mismatch for {key}: "
            f"got {len(ids)}, expected {condition.fewshot_k}"
        )
    out: list[dict] = []
    for iid in ids:
        if iid not in by_id:
            raise RuntimeError(
                f"Fewshot item {iid!r} not found in {benchmark.benchmark_id} "
                f"split={src_split}"
            )
        out.append(by_id[iid])
    return out


def _build_condition_context(
    condition: ConditionRow,
    merged: MergedConfig,
    fewshot_lookup: dict,
    eval_pool_cache: dict[tuple[str, str], list[dict]],
    fewshot_pool_cache: dict[tuple[str, str], list[dict]],
) -> ConditionContext:
    benchmark = next(b for b in merged.benchmarks if b.benchmark_id == condition.benchmark_id)
    prompt_spec = next(
        p for p in merged.prompts
        if p.benchmark_id == condition.benchmark_id and p.prompt_id == condition.prompt_id
    )
    scoring_rule = next(
        s for s in merged.scoring_rules if s.scoring_rule_id == condition.scoring_rule_id
    )
    template_text, template_hash = _load_template(prompt_spec)

    eval_key = (benchmark.benchmark_id, benchmark.eval_split)
    if eval_key not in eval_pool_cache:
        eval_pool_cache[eval_key] = hf_load_split(benchmark, benchmark.eval_split)
    eval_items = filter_to_subject(eval_pool_cache[eval_key], condition.subject_id)

    fewshot_items = _resolve_fewshot_items(
        condition, benchmark, fewshot_lookup, fewshot_pool_cache,
    )
    return ConditionContext(
        benchmark=benchmark,
        prompt_spec=prompt_spec,
        scoring_rule=scoring_rule,
        template_text=template_text,
        template_hash=template_hash,
        eval_items=eval_items,
        fewshot_items=fewshot_items,
        fewshot_k=condition.fewshot_k,
    )


def _score_one_item(
    model: Any,
    tokenizer: Any,
    item: dict,
    ctx: ConditionContext,
    condition: ConditionRow,
) -> dict[str, Any]:
    rendered = render_prompt(
        ctx.template_text, item, ctx.fewshot_items, ctx.benchmark, ctx.prompt_spec
    )
    rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    base = {
        "condition_id": condition.condition_id,
        "item_id": get_native_id(item, ctx.benchmark),
        "subject_id": get_subject(item, condition.benchmark_id),
        "prompt_text_hash": ctx.template_hash,
        "rendered_prompt_hash": rendered_hash,
        "scoring_rule_id": condition.scoring_rule_id,
    }

    if ctx.scoring_rule.scoring_rule_id == "ll_norm":
        choices = get_choices(item, condition.benchmark_id)
        gold_letter = get_gold_letter(item, condition.benchmark_id)
        ll = score_ll_norm(model, tokenizer, rendered, len(choices))
        is_correct = (ll["selected_answer"] == gold_letter)
        return {
            **base,
            "gold_answer": gold_letter,
            "choices": ll["choices"],
            "choice_scores": ll["choice_scores"],
            "choice_token_counts": ll["choice_token_counts"],
            "normalization": ll["normalization"],
            "selected_answer": ll["selected_answer"],
            "is_correct": is_correct,
        }

    # Generate-and-parse path.
    is_open_gen = (ctx.benchmark.task_type == "open_generation")
    if is_open_gen:
        gold = get_gold_text(item, condition.benchmark_id)
        n_choices = None
        max_new = int(ctx.scoring_rule.generation_params["max_new_tokens"])
    else:
        gold = get_gold_letter(item, condition.benchmark_id)
        n_choices = len(get_choices(item, condition.benchmark_id))
        max_new = int(ctx.scoring_rule.generation_params["max_new_tokens"])

    gp = score_generate_parse(
        model, tokenizer, rendered,
        parse_strategy=ctx.scoring_rule.parse_strategy,
        n_choices=n_choices,
        max_new_tokens=max_new,
    )
    is_correct = (gp["parse_status"] == "parsed" and gp["parsed_answer"] == gold)
    return {
        **base,
        "gold_answer": gold,
        "raw_generation": gp["raw_generation"],
        "parsed_answer": gp["parsed_answer"],
        "parse_status": gp["parse_status"],
        "is_correct": is_correct,
        "generation_params": {
            "decoding": "greedy",
            "temperature": 0,
            "max_new_tokens": max_new,
        },
        "_generation_length_tokens": gp["generation_length_tokens"],  # not in schema; passed through
    }


# ---------------------------------------------------------------------------
# Output writing (atomic .tmp/ + rename)
# ---------------------------------------------------------------------------


def _utc_iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _drop_underscore_keys(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _write_atomic(
    out_dir: Path,
    condition: ConditionRow,
    ctx: ConditionContext,
    rows: list[dict],
    started_at: float,
    finished_at: float,
    schemas: dict[str, Any],
    model_spec: ModelSpec,
    transformers_version: str,
    torch_version: str,
) -> None:
    """Write run_metadata.json + jsonl atomically. Validates before rename."""
    import jsonschema

    tmp_root = out_dir / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / condition.condition_id
    tmp_dir = tmp_root / condition.condition_id
    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True)

    is_ll = (condition.scoring_rule_id == "ll_norm")
    if is_ll:
        jsonl_name = "item_scores.jsonl"
        row_schema = schemas["item_scores"]
    else:
        jsonl_name = "item_outputs.jsonl"
        row_schema = schemas["item_outputs"]

    jsonl_path = tmp_dir / jsonl_name
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            clean = _drop_underscore_keys(row)
            jsonschema.validate(clean, row_schema)
            f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    metadata = {
        "run_id": condition.condition_id,
        "condition_id": condition.condition_id,
        "model_id": model_spec.model_id,
        "hf_name": model_spec.hf_name,
        "hf_revision": model_spec.hf_revision,
        "benchmark_id": condition.benchmark_id,
        "subject_id": condition.subject_id,
        "prompt_id": condition.prompt_id,
        "seed_id": condition.seed_id,
        "fewshot_k": condition.fewshot_k,
        "scoring_rule_id": condition.scoring_rule_id,
        "task_type": condition.task_type,
        "num_items_expected": condition.expected_num_items,
        "num_items_completed": len(rows),
        "inference_backend": INFERENCE_BACKEND,
        "python_version": platform.python_version(),
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "lm_eval_version": None,
        "device": "cuda",
        "dtype": "bfloat16",
        "started_at": _utc_iso(started_at),
        "finished_at": _utc_iso(finished_at),
        "status": (
            "complete" if len(rows) == condition.expected_num_items else "partial"
        ),
        "notes": "",
    }
    jsonschema.validate(metadata, schemas["run_metadata"])
    (tmp_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if final_dir.exists():
        # Should not happen due to resume-check, but guard anyway.
        raise RuntimeError(f"Final dir already exists: {final_dir}")
    os.replace(tmp_dir, final_dir)


def _log_failure(out_dir: Path, condition_id: str, exc: BaseException) -> None:
    fail_path = out_dir.parent / "failures.jsonl"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "condition_id": condition_id,
        "exception_class": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "timestamp": _utc_iso(time.time()),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pre-flight (§9)
# ---------------------------------------------------------------------------


def run_preflight(
    merged: MergedConfig,
    conditions: list[ConditionRow],
    fewshot_lookup: dict,
    out_dir: Path,
    schemas: dict[str, Any],
) -> None:
    print("[preflight] checking schemas readable…")
    for k, sch in schemas.items():
        if not isinstance(sch, dict) or "type" not in sch:
            raise RuntimeError(f"Schema {k!r} missing or malformed")

    print("[preflight] checking prompt files + content hashes…")
    needed = {(c.benchmark_id, c.prompt_id) for c in conditions}
    for prompt_spec in merged.prompts:
        if (prompt_spec.benchmark_id, prompt_spec.prompt_id) not in needed:
            continue
        _load_template(prompt_spec)  # raises on hash mismatch

    print("[preflight] checking output dir writable…")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".tmp").mkdir(exist_ok=True)
    probe = out_dir / ".tmp" / "_writable_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    print("[preflight] checking fewshot lock entries cover all conditions…")
    for c in conditions:
        if c.fewshot_k == 0:
            continue
        key = (c.benchmark_id, c.subject_id, _normalize_seed(c.seed_id))
        if key not in fewshot_lookup:
            raise RuntimeError(f"Fewshot manifest missing entry for {key}")

    print("[preflight] checking models loadable (1 model probe)…")
    primary_models = [m for m in merged.models if m.primary]
    if not primary_models:
        raise RuntimeError("No primary models in models.yaml")
    probe_model = primary_models[0]
    print(f"[preflight] probing {probe_model.hf_name}@{probe_model.hf_revision[:8]}")
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(probe_model.hf_name, revision=probe_model.hf_revision)

    print("[preflight] OK")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _load_schemas(schemas_dir: Path) -> dict[str, Any]:
    return {
        "run_metadata": json.loads((schemas_dir / "run_metadata.schema.json").read_text()),
        "item_outputs": json.loads((schemas_dir / "item_outputs.schema.json").read_text()),
        "item_scores": json.loads((schemas_dir / "item_scores.schema.json").read_text()),
    }


def _filter_pending(conditions: list[ConditionRow], out_dir: Path) -> list[ConditionRow]:
    pending: list[ConditionRow] = []
    for c in conditions:
        meta = out_dir / c.condition_id / "run_metadata.json"
        if meta.exists():
            continue
        if c.condition_status not in {"pending", "running", "failed"}:
            continue
        pending.append(c)
    return pending


def _print_summary(conditions: list[ConditionRow], out_dir: Path) -> None:
    completed = sum(
        1 for c in conditions
        if (out_dir / c.condition_id / "run_metadata.json").exists()
    )
    failures_path = out_dir.parent / "failures.jsonl"
    n_fail = 0
    if failures_path.exists():
        n_fail = sum(1 for _ in open(failures_path, "r", encoding="utf-8"))
    pending = len(conditions) - completed
    print(
        f"[summary] completed={completed}/{len(conditions)} "
        f"pending={pending} failures={n_fail}"
    )


def _run_condition(
    condition: ConditionRow,
    model: Any,
    tokenizer: Any,
    merged: MergedConfig,
    fewshot_lookup: dict,
    eval_pool_cache: dict,
    fewshot_pool_cache: dict,
    schemas: dict,
    out_dir: Path,
    transformers_version: str,
    torch_version: str,
    model_spec: ModelSpec,
) -> None:
    started = time.time()
    ctx = _build_condition_context(
        condition, merged, fewshot_lookup, eval_pool_cache, fewshot_pool_cache,
    )

    rows: list[dict] = []
    for it in ctx.eval_items:
        try:
            row = _score_one_item(model, tokenizer, it, ctx, condition)
        except Exception as item_exc:
            row = {
                "condition_id": condition.condition_id,
                "item_id": get_native_id(it, ctx.benchmark),
                "subject_id": get_subject(it, condition.benchmark_id),
                "prompt_text_hash": ctx.template_hash,
                "rendered_prompt_hash": "0" * 64,
                "scoring_rule_id": condition.scoring_rule_id,
                "gold_answer": "",
                "raw_generation": f"<runtime_error: {type(item_exc).__name__}: {item_exc}>",
                "parsed_answer": None,
                "parse_status": "runtime_error",
                "is_correct": False,
                "generation_params": {"decoding": "greedy", "temperature": 0, "max_new_tokens": 0},
            }
            print(
                f"[item-error] {condition.condition_id} item={row['item_id']} "
                f"{type(item_exc).__name__}: {item_exc}"
            )
        rows.append(row)

    finished = time.time()
    _write_atomic(
        out_dir, condition, ctx, rows, started, finished,
        schemas, model_spec, transformers_version, torch_version,
    )
    n = len(rows)
    elapsed = finished - started
    rate = n / elapsed if elapsed > 0 else 0.0
    print(
        f"[done] {condition.condition_id}: {n} items, "
        f"{elapsed:.1f}s, {rate:.1f} items/s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU-side inference driver")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--conditions", type=Path, default=None)
    parser.add_argument("--conditions-secondary", type=Path, default=None)
    parser.add_argument("--fewshot", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--filter-model", default=None,
                        help="Only run conditions for this model_id")
    parser.add_argument("--filter-benchmark", default=None,
                        help="Only run conditions for this benchmark_id")
    parser.add_argument("--max-conditions", type=int, default=None,
                        help="Run at most N pending conditions and exit")
    args = parser.parse_args(argv)

    merged = load_config(args.config)
    paper_root = args.config.resolve().parent.parent
    schemas_dir = paper_root / merged.study_config.paths.get("schemas_dir", "schemas/")
    out_dir = args.out_dir or (paper_root / merged.study_config.paths.get("runs_raw_dir", "runs/raw/"))
    out_dir = Path(out_dir)

    schemas = _load_schemas(Path(schemas_dir))

    cond_path_primary = args.conditions or (paper_root / "manifests/conditions_primary.csv")
    cond_path_secondary = args.conditions_secondary or (
        paper_root / "manifests/conditions_gsm8k.csv"
    )
    fewshot_path = args.fewshot or (paper_root / "manifests/fewshot_manifest.csv")

    conditions: list[ConditionRow] = []
    if cond_path_primary.exists():
        conditions.extend(load_conditions(cond_path_primary))
    if cond_path_secondary.exists():
        conditions.extend(load_conditions(cond_path_secondary))
    fewshot_lookup = load_fewshot_manifest(fewshot_path) if fewshot_path.exists() else {}

    if args.filter_model:
        conditions = [c for c in conditions if c.model_id == args.filter_model]
    if args.filter_benchmark:
        conditions = [c for c in conditions if c.benchmark_id == args.filter_benchmark]

    print(f"[main] {len(conditions)} conditions in scope; out_dir={out_dir}")

    if args.preflight_only:
        run_preflight(merged, conditions, fewshot_lookup, out_dir, schemas)
        return 0

    run_preflight(merged, conditions, fewshot_lookup, out_dir, schemas)

    pending = _filter_pending(conditions, out_dir)
    if args.max_conditions is not None:
        pending = pending[: args.max_conditions]
    print(f"[main] {len(pending)} pending after resume-skip")

    if not pending:
        _print_summary(conditions, out_dir)
        return 0

    # Group by model_id; load each model exactly once.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Group preserving manifest order.
    groups: dict[str, list[ConditionRow]] = {}
    for c in pending:
        groups.setdefault(c.model_id, []).append(c)

    eval_pool_cache: dict[tuple[str, str], list[dict]] = {}
    fewshot_pool_cache: dict[tuple[str, str], list[dict]] = {}

    for model_id, group in groups.items():
        model_spec = next(m for m in merged.models if m.model_id == model_id)
        print(
            f"[load] {model_spec.hf_name}@{model_spec.hf_revision[:8]} "
            f"({len(group)} conditions)"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec.hf_name, revision=model_spec.hf_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.hf_name,
            revision=model_spec.hf_revision,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        try:
            for i, condition in enumerate(group):
                idx_global = i + 1
                print(
                    f"[run] [{idx_global}/{len(group)}] {condition.condition_id}"
                )
                try:
                    _run_condition(
                        condition, model, tokenizer, merged,
                        fewshot_lookup, eval_pool_cache, fewshot_pool_cache,
                        schemas, out_dir,
                        transformers.__version__, torch.__version__, model_spec,
                    )
                except Exception as exc:
                    _log_failure(out_dir, condition.condition_id, exc)
                    print(
                        f"[fail] {condition.condition_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        finally:
            del model
            del tokenizer
            torch.cuda.empty_cache()

    _print_summary(conditions, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
