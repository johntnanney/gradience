"""make_fixtures.py — SPEC_CPU_v0_2 §16.6

Generate test fixtures deterministically. Idempotent: re-running produces
byte-identical output (modulo trailing-newline normalization).

Fixtures produced:
    tiny_item_outputs.jsonl         — 4-item valid G&P fixture
    tiny_item_scores.jsonl          — 4-item valid LL fixture
    tiny_run_metadata_gp.json       — valid G&P run metadata
    tiny_run_metadata_ll.json       — valid LL run metadata
    tiny_condition_scores.csv       — 4-row condition-level fixture
    pathological_all_parse_fail.jsonl
    pathological_near_tied_ll.jsonl
    configs/*.yaml                  — minimal-but-valid fixture configs

Usage:
    python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


# --- Helpers ----------------------------------------------------------------


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            json.dump(r, f, sort_keys=True)
            f.write("\n")


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --- Tiny valid G&P fixture -------------------------------------------------


def make_tiny_item_outputs() -> list[dict]:
    """Small valid item_outputs.jsonl for constrained-choice G&P scoring.

    4 items, 2 correct and 2 incorrect; all parseable.
    """
    condition_id = "arc_challenge__pythia_410m__P1_original__s42__generate_parse"
    p_hash = sha256_str("PROMPT_TEMPLATE_V1")
    data = [
        ("B", "The answer is B.", "B", True),
        ("A", "The answer is A.", "A", True),
        ("C", "The answer is B.", "B", False),
        ("D", "I think it is A.", "A", False),
    ]
    rows = []
    for i, (gold, gen, parsed, correct) in enumerate(data, start=1):
        rows.append({
            "condition_id": condition_id,
            "item_id": f"arc-c-test-{i:03d}",
            "subject_id": None,
            "prompt_text_hash": p_hash,
            "rendered_prompt_hash": sha256_str(f"ITEM_{i}_RENDERED"),
            "gold_answer": gold,
            "raw_generation": gen,
            "parsed_answer": parsed,
            "parse_status": "parsed",
            "is_correct": correct,
            "scoring_rule_id": "generate_parse",
            "generation_params": {
                "decoding": "greedy",
                "temperature": 0,
                "max_new_tokens": 32,
            },
        })
    return rows


# --- Tiny valid LL fixture --------------------------------------------------


def make_tiny_item_scores() -> list[dict]:
    condition_id = "arc_challenge__pythia_410m__P1_original__s42__ll_norm"
    p_hash = sha256_str("PROMPT_TEMPLATE_V1")
    # Each tuple: (gold, scores dict). Selected = argmax(scores).
    data = [
        ("B", {"A": -2.44, "B": -1.92, "C": -2.31, "D": -3.10}),  # B selected, correct
        ("A", {"A": -1.50, "B": -2.80, "C": -2.90, "D": -3.00}),  # A selected, correct
        ("C", {"A": -2.00, "B": -2.50, "C": -2.60, "D": -2.70}),  # A selected, wrong
        ("D", {"A": -3.00, "B": -3.10, "C": -3.20, "D": -3.30}),  # A selected, wrong
    ]
    choices = ["A", "B", "C", "D"]
    rows = []
    for i, (gold, scores) in enumerate(data, start=1):
        selected = max(scores, key=lambda k: scores[k])
        rows.append({
            "condition_id": condition_id,
            "item_id": f"arc-c-test-{i:03d}",
            "subject_id": None,
            "prompt_text_hash": p_hash,
            "rendered_prompt_hash": sha256_str(f"ITEM_{i}_LL_RENDERED"),
            "gold_answer": gold,
            "choices": choices,
            "choice_scores": scores,
            "choice_token_counts": {"A": 4, "B": 5, "C": 4, "D": 6},
            "normalization": "length_normalized",
            "selected_answer": selected,
            "is_correct": (selected == gold),
            "scoring_rule_id": "ll_norm",
        })
    return rows


# --- Tiny run metadata ------------------------------------------------------


def _base_run_metadata(condition_id: str, scoring_rule_id: str,
                       num_items: int) -> dict:
    return {
        "run_id": condition_id,
        "condition_id": condition_id,
        "model_id": "pythia_410m",
        "hf_name": "EleutherAI/pythia-410m",
        "hf_revision": "fixture_revision_hash",
        "benchmark_id": "arc_challenge",
        "subject_id": None,
        "prompt_id": "P1_original",
        "seed_id": "42",
        "fewshot_k": 5,
        "scoring_rule_id": scoring_rule_id,
        "task_type": "constrained_choice",
        "num_items_expected": num_items,
        "num_items_completed": num_items,
        "inference_backend": "local_transformers",
        "python_version": "3.11.6",
        "transformers_version": "4.44.0",
        "torch_version": "2.1.0",
        "lm_eval_version": None,
        "device": "cpu",
        "dtype": "float32",
        "started_at": "2026-04-24T00:00:00+00:00",
        "finished_at": "2026-04-24T00:00:10+00:00",
        "status": "complete",
        "notes": "",
    }


def make_tiny_run_metadata_gp() -> dict:
    return _base_run_metadata(
        "arc_challenge__pythia_410m__P1_original__s42__generate_parse",
        "generate_parse",
        4,
    )


def make_tiny_run_metadata_ll() -> dict:
    return _base_run_metadata(
        "arc_challenge__pythia_410m__P1_original__s42__ll_norm",
        "ll_norm",
        4,
    )


# --- Tiny condition-level CSV fixture ---------------------------------------


def make_tiny_condition_scores() -> list[dict]:
    """Small valid condition_level CSV: 4 conditions across 2 models/prompts."""
    return [
        {
            "condition_id": "arc_challenge__pythia_410m__P1_original__s42__generate_parse",
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_410m",
            "prompt_id": "P1_original",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "num_items": 4,
            "num_correct": 2,
            "accuracy": 0.5,
            "parseability_rate": 1.0,
            "mean_generation_length_tokens": 5.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "arc_challenge__pythia_410m__P2_lm_eval__s42__generate_parse",
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_410m",
            "prompt_id": "P2_lm_eval",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "num_items": 4,
            "num_correct": 3,
            "accuracy": 0.75,
            "parseability_rate": 1.0,
            "mean_generation_length_tokens": 6.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "arc_challenge__pythia_1_4b__P1_original__s42__generate_parse",
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_1_4b",
            "prompt_id": "P1_original",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "num_items": 4,
            "num_correct": 3,
            "accuracy": 0.75,
            "parseability_rate": 1.0,
            "mean_generation_length_tokens": 5.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "arc_challenge__pythia_1_4b__P2_lm_eval__s42__generate_parse",
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_1_4b",
            "prompt_id": "P2_lm_eval",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "num_items": 4,
            "num_correct": 4,
            "accuracy": 1.0,
            "parseability_rate": 1.0,
            "mean_generation_length_tokens": 6.0,
            "condition_status": "complete",
        },
    ]


# --- Pathological fixtures --------------------------------------------------


def make_pathological_all_parse_fail() -> list[dict]:
    """G&P condition where every item fails to parse.

    Exercises the SPEC §5.3 rule that parse failures score as incorrect
    but parseability_rate is reported separately. Five items span all
    five non-parsed parse_status values.
    """
    condition_id = "test_bench__test_model__P1__s42__generate_parse"
    p_hash = sha256_str("PATHOLOGICAL_V1")
    statuses = [
        "unparseable",
        "empty",
        "multiple_answers",
        "invalid_choice",
        "runtime_error",
    ]
    rows = []
    for i, status in enumerate(statuses, start=1):
        rows.append({
            "condition_id": condition_id,
            "item_id": f"test-item-{i:03d}",
            "subject_id": None,
            "prompt_text_hash": p_hash,
            "rendered_prompt_hash": sha256_str(f"PATH_{i}"),
            "gold_answer": "B",
            "raw_generation": "" if status == "empty" else "garbage output",
            "parsed_answer": None,
            "parse_status": status,
            "is_correct": False,
            "scoring_rule_id": "generate_parse",
            "generation_params": {
                "decoding": "greedy",
                "temperature": 0,
                "max_new_tokens": 32,
            },
        })
    return rows


def make_pathological_near_tied_ll() -> list[dict]:
    """LL condition with near-tied top-2 choices.

    Exercises ll_margin_top2 computation; differences between top-1 and
    top-2 are smaller than 1e-3, near the threshold where floating-point
    path differences can flip the selected answer.
    """
    condition_id = "test_bench__test_model__P1__s42__ll_norm"
    p_hash = sha256_str("TIED_V1")
    data = [
        ("A", {"A": -2.0000001, "B": -2.0000002, "C": -5.0, "D": -5.0}),
        ("B", {"A": -1.9999,    "B": -1.9998,    "C": -5.0, "D": -5.0}),
        ("A", {"A": -3.0,       "B": -3.0000001, "C": -5.0, "D": -5.0}),
    ]
    choices = ["A", "B", "C", "D"]
    rows = []
    for i, (gold, scores) in enumerate(data, start=1):
        selected = max(scores, key=lambda k: scores[k])
        rows.append({
            "condition_id": condition_id,
            "item_id": f"tied-item-{i:03d}",
            "subject_id": None,
            "prompt_text_hash": p_hash,
            "rendered_prompt_hash": sha256_str(f"TIED_{i}"),
            "gold_answer": gold,
            "choices": choices,
            "choice_scores": scores,
            "choice_token_counts": {"A": 4, "B": 5, "C": 4, "D": 6},
            "normalization": "length_normalized",
            "selected_answer": selected,
            "is_correct": (selected == gold),
            "scoring_rule_id": "ll_norm",
        })
    return rows


# --- Minimal-but-valid fixture configs --------------------------------------


def _make_prompts_yaml() -> str:
    out = ["prompts:"]
    benchmarks = [f"test_bench_{i}" for i in range(1, 6)]
    prompt_ids = [
        "P1_original",
        "P2_lm_eval",
        "P3_helm_or_published",
        "P4_minimal_sourced",
    ]
    source_types = [
        "benchmark_authors",
        "lm_eval_harness",
        "helm_reference",
        "author_minimal_declared",
    ]
    for b in benchmarks:
        for pid, src in zip(prompt_ids, source_types):
            out.append(
                f"  - benchmark_id: {b}\n"
                f"    prompt_id: {pid}\n"
                f"    prompt_path: prompts/{b}/{pid}.txt\n"
                f"    content_hash: TODO\n"
                f"    source_type: {src}\n"
                f"    source_url_or_ref: test://source/{b}/{pid}\n"
                f"    source_commit: null\n"
                f"    admissibility_status: draft\n"
                f"    placeholders_used:\n"
                f"      - question\n"
                f"      - answer_instruction\n"
                f"    notes: test fixture prompt"
            )
    return "\n".join(out) + "\n"


def make_minimal_config_files(target_dir: Path) -> None:
    """Write minimal-but-valid YAML configs for config-loader tests."""
    target_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "study_config.yaml").write_text("""\
study_id: test_study
prereg_version: v1_test
tier_primary_enabled: true
tier_secondary_enabled: false
extension_7b_enabled: false
paths:
  configs_dir: configs/
  manifests_dir: manifests/
  prompts_dir: prompts/
  runs_raw_dir: runs/raw/
  runs_normalized_dir: runs/normalized/
  analysis_dir: analysis/
  reports_dir: reports/
  schemas_dir: schemas/
includes:
  - configs/models.yaml
  - configs/benchmarks.yaml
  - configs/prompts.yaml
  - configs/scoring_rules.yaml
  - configs/analysis_config.yaml
""", encoding="utf-8")

    (target_dir / "models.yaml").write_text("""\
models:
  - model_id: test_model_1
    hf_name: test/model-1
    hf_revision: abc123
    model_family: test_family
    model_type: base
    parameter_count_b: 0.1
    primary: true
    optional_extension: false
  - model_id: test_model_2
    hf_name: test/model-2
    hf_revision: def456
    model_family: test_family
    model_type: base
    parameter_count_b: 0.2
    primary: true
    optional_extension: false
  - model_id: test_model_3
    hf_name: test/model-3
    hf_revision: ghi789
    model_family: other_family
    model_type: instruct
    parameter_count_b: 0.3
    primary: true
    optional_extension: false
""", encoding="utf-8")

    _benchmark_yaml_parts = []
    _benchmark_yaml_parts.append("benchmarks:")
    for i in range(1, 6):
        seed_facet = "true" if i != 3 else "false"
        fewshot_k = 5 if i != 3 else 0
        source_split = "train" if i != 3 else "null"
        eval_split = "test" if i == 1 else "validation"
        _benchmark_yaml_parts.append(
            f"  - benchmark_id: test_bench_{i}\n"
            f"    display_name: Test Benchmark {i}\n"
            f"    tier: primary\n"
            f"    hf_dataset: test/bench{i}\n"
            f"    hf_config: null\n"
            f"    dataset_version_hash: hash{i}\n"
            f"    task_type: constrained_choice\n"
            f"    fewshot_k: {fewshot_k}\n"
            f"    seed_facet: {seed_facet}\n"
            f"    fewshot_source_split: {source_split}\n"
            f"    eval_split: {eval_split}\n"
            f"    scoring_rules: [ll_norm, generate_parse]\n"
            f"    item_id_field: id\n"
            f"    answer_field: answer\n"
            f"    expected_num_items: 10"
        )
    (target_dir / "benchmarks.yaml").write_text("\n".join(_benchmark_yaml_parts) + "\n",
                                                 encoding="utf-8")

    (target_dir / "prompts.yaml").write_text(_make_prompts_yaml(), encoding="utf-8")

    (target_dir / "scoring_rules.yaml").write_text("""\
scoring_rules:
  - scoring_rule_id: ll_norm
    display_name: Length-Normalized Log-Likelihood
    applies_to_task_types: [constrained_choice]
    normalization: length_normalized
    selection_rule: argmax
  - scoring_rule_id: generate_parse
    display_name: Generate and Regex-Parse
    applies_to_task_types: [constrained_choice]
    parse_strategy: benchmark_specific_regex
""", encoding="utf-8")

    (target_dir / "analysis_config.yaml").write_text("""\
bootstrap:
  n_resamples: 100
  random_seed: 42
  ci_lower_percentile: 2.5
  ci_upper_percentile: 97.5
mixed_effects_cascade:
  level_1:
    random_effects: [prompt, seed, scoring_rule, item]
  level_2:
    random_effects: [prompt, seed, scoring_rule, item]
  level_3:
    random_effects: [prompt, scoring_rule, item]
  level_4:
    method: aggregate_g_theory_only
convergence_triggers:
  max_singular_warning: trigger_fallback
  hessian_non_pd: trigger_fallback
  iteration_limit_exceeded: trigger_fallback
  gradient_norm_above: 1.0e-3
tolerance:
  decision_rule_threshold_h1: 0.005
  benchmarks_required_for_h1: 3
h2_generalizability_threshold: 0.80
h3_ranking_reversal_threshold: 0.10
h4_mmlu_interaction_threshold: 0.10
""", encoding="utf-8")


# --- Mock analysis-output fixtures (for 98_/99_ reporting tests) ------------


def make_mock_variance_components_csv() -> list[dict]:
    """Minimal aggregate-VC rows for 98_reproducibility_trace testing.

    Mimics the schema documented in SPEC §8.7:
        benchmark_id, model_id, source, variance, proportion, df.
    Two benchmarks x two models x three sources, with toy values whose
    proportions sum to 1.0 within each (benchmark_id, model_id) block.
    """
    rows: list[dict] = []
    for bench in ("test_bench_1", "test_bench_2"):
        for model in ("test_model_1", "test_model_2"):
            rows.append({
                "benchmark_id": bench, "model_id": model,
                "source": "var_prompt",
                "variance": 0.0010, "proportion": 0.5, "df": 3,
            })
            rows.append({
                "benchmark_id": bench, "model_id": model,
                "source": "var_seed",
                "variance": 0.0004, "proportion": 0.2, "df": 2,
            })
            rows.append({
                "benchmark_id": bench, "model_id": model,
                "source": "var_residual",
                "variance": 0.0006, "proportion": 0.3, "df": 18,
            })
    return rows


def make_mock_tolerance_schedule_csv() -> list[dict]:
    """Minimal tolerance-by-cell rows for reporting tests.

    Mimics a subset of SPEC §8.8's tolerance_by_cell.csv columns. The
    point estimates and CIs are fabricated but internally coherent
    (lower < point < upper).
    """
    rows: list[dict] = []
    for bench in ("test_bench_1", "test_bench_2"):
        for model in ("test_model_1", "test_model_2"):
            for rule in ("ll_norm", "generate_parse"):
                rows.append({
                    "benchmark_id": bench,
                    "model_id": model,
                    "scoring_rule_id": rule,
                    "sem_single": 0.012,
                    "tolerance_single": 0.024,
                    "tolerance_single_ci_lower": 0.018,
                    "tolerance_single_ci_upper": 0.032,
                    "sem_full_design": 0.0024,
                    "tolerance_full_design": 0.0048,
                    "tolerance_full_design_ci_lower": 0.0036,
                    "tolerance_full_design_ci_upper": 0.0064,
                    "licensed_precision_single": "interval_required",
                    "licensed_precision_full_design": "two_decimal_accuracy",
                })
    return rows


def make_mock_tolerance_by_benchmark_summary_csv() -> list[dict]:
    """Cross-model-median tolerance per benchmark (SPEC §8.8)."""
    return [
        {
            "benchmark_id": "test_bench_1",
            "tolerance_single_median": 0.024,
            "tolerance_single_median_ci_lower": 0.019,
            "tolerance_single_median_ci_upper": 0.030,
            "n_models": 3,
        },
        {
            "benchmark_id": "test_bench_2",
            "tolerance_single_median": 0.022,
            "tolerance_single_median_ci_lower": 0.017,
            "tolerance_single_median_ci_upper": 0.028,
            "n_models": 3,
        },
    ]


def make_mock_h1_test_json() -> dict:
    """Minimal h1_test.json payload (SPEC §8.8)."""
    return {
        "h1_hypothesis": (
            "bootstrap lower bound of cross-model median single-occasion "
            "tolerance > 0.005 for at least 3 of 5 benchmarks"
        ),
        "threshold": 0.005,
        "n_required": 3,
        "n_exceeding": 3,
        "benchmarks_exceeding": [
            "arc_challenge", "hellaswag", "mmlu_panel",
        ],
        "h1_confirmed": True,
    }


def make_mock_h4_test_json() -> dict:
    """Minimal MMLU h4_test.json payload (SPEC §8.10)."""
    return {
        "h4_hypothesis": (
            "MMLU model x subject interaction proportion > 0.10 of "
            "total variance"
        ),
        "threshold": 0.10,
        "model_x_subject_proportion": 0.18,
        "h4_confirmed": True,
    }


def write_mock_analysis_outputs(target_dir: Path) -> None:
    """Write the mock analysis tree under target_dir.

    Layout produced (subset of SPEC §8.7-§8.10):

        target_dir/
          variance_components/aggregate_vc.csv
          tolerance_schedules/tolerance_by_cell.csv
          tolerance_schedules/tolerance_by_benchmark_summary.csv
          tolerance_schedules/h1_test.json
          mmlu_subjects/h4_test.json
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    vc_dir = target_dir / "variance_components"
    vc_dir.mkdir(parents=True, exist_ok=True)
    vc_rows = make_mock_variance_components_csv()
    write_csv(vc_dir / "aggregate_vc.csv", vc_rows, list(vc_rows[0].keys()))

    tol_dir = target_dir / "tolerance_schedules"
    tol_dir.mkdir(parents=True, exist_ok=True)
    tol_rows = make_mock_tolerance_schedule_csv()
    write_csv(tol_dir / "tolerance_by_cell.csv", tol_rows,
              list(tol_rows[0].keys()))
    bench_summary = make_mock_tolerance_by_benchmark_summary_csv()
    write_csv(tol_dir / "tolerance_by_benchmark_summary.csv",
              bench_summary, list(bench_summary[0].keys()))
    write_json(tol_dir / "h1_test.json", make_mock_h1_test_json())

    mmlu_dir = target_dir / "mmlu_subjects"
    mmlu_dir.mkdir(parents=True, exist_ok=True)
    write_json(mmlu_dir / "h4_test.json", make_mock_h4_test_json())


# --- MMLU subject-decomp item-level fixture --------------------------------


def make_mmlu_fixture_item_level() -> list[dict]:
    """Synthetic MMLU-panel item-level rows for decomposition testing.

    3 models x 5 subjects x 2 prompts x 3 items_per_subject = 90 rows.

    Designed so that model x subject interaction is nontrivial: each
    model has a different "preferred subject" — its accuracy on that
    subject is high while its accuracy on other subjects is moderate.
    The per-subject means alone do not capture this; only the
    model-by-subject interaction term does. This guarantees the H4
    threshold check has a non-degenerate proportion to evaluate.
    """
    models = ["test_model_1", "test_model_2", "test_model_3"]
    subjects = [
        "world_religions",
        "high_school_mathematics",
        "psychology",
        "professional_medicine",
        "global_facts",
    ]
    prompts = ["P1_original", "P2_lm_eval"]
    items_per_subject = 3

    # Each model's "preferred" subject — accuracy is reliably high there,
    # moderate elsewhere. The deterministic correctness pattern below is
    # chosen so that subject means are similar, model means are similar,
    # but the model x subject interaction has a sizeable share.
    preferred = {
        "test_model_1": "world_religions",
        "test_model_2": "high_school_mathematics",
        "test_model_3": "psychology",
    }
    # For two of the five subjects (professional_medicine, global_facts),
    # no model has a preference; this introduces some shared variance.

    rows: list[dict] = []
    for model in models:
        for subject in subjects:
            for prompt in prompts:
                for item_ix in range(items_per_subject):
                    item_id = f"{subject}-{item_ix:03d}"
                    is_pref = (preferred.get(model) == subject)
                    base_correct = (item_ix + (1 if prompt == "P2_lm_eval" else 0)) % 2 == 0
                    if is_pref:
                        is_correct = True
                    elif subject in ("professional_medicine", "global_facts"):
                        is_correct = base_correct
                    else:
                        is_correct = (item_ix == 0)
                    cond_id = (
                        f"mmlu_panel__{model}__{prompt}__s42__ll_norm__{subject}"
                    )
                    rows.append({
                        "condition_id": cond_id,
                        "tier": "primary",
                        "benchmark_id": "mmlu_panel",
                        "subject_id": subject,
                        "model_id": model,
                        "model_family": "test_family",
                        "model_type": "base",
                        "parameter_count_b": 0.1,
                        "prompt_id": prompt,
                        "prompt_source_type": "benchmark_authors",
                        "seed_id": "42",
                        "fewshot_k": 5,
                        "scoring_rule_id": "ll_norm",
                        "task_type": "constrained_choice",
                        "item_id": item_id,
                        "gold_answer": "A",
                        "predicted_answer": "A" if is_correct else "B",
                        "is_correct": is_correct,
                        "parse_status": None,
                        "parseable": None,
                        "choice_count": 4,
                        "prompt_text_hash": sha256_str(f"PROMPT/{prompt}"),
                        "rendered_prompt_hash": sha256_str(f"R/{model}/{subject}/{item_ix}"),
                        "run_id": cond_id,
                        "selected_answer": "A" if is_correct else "B",
                        "ll_margin_top2": 0.5,
                        "gold_choice_score": -1.0,
                        "selected_choice_score": -1.0,
                        "raw_generation": None,
                        "parsed_answer": None,
                        "generation_length_tokens": None,
                    })
    return rows


def write_mmlu_fixture_parquet(path: Path) -> None:
    """Write make_mmlu_fixture_item_level() to a parquet file."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = make_mmlu_fixture_item_level()
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# --- GSM8K case-study item-level fixture -----------------------------------


def make_gsm8k_fixture_item_level() -> list[dict]:
    """Synthetic GSM8K item-level rows with two scoring rules.

    2 models x 2 prompts x 2 seeds x 2 scoring rules x 4 items = 64 rows.

    Constructed so that permissive parsing accepts more answers than
    strict parsing for the same items, exercising the extraction
    sensitivity delta.
    """
    models = ["test_model_1", "test_model_2"]
    prompts = ["P1_original", "P2_lm_eval"]
    seeds = ["42", "43"]
    rules = ["generate_parse_strict", "generate_parse_permissive"]
    n_items = 4

    rows: list[dict] = []
    for model in models:
        for prompt in prompts:
            for seed in seeds:
                for rule in rules:
                    cond_id = f"gsm8k__{model}__{prompt}__s{seed}__{rule}"
                    for item_ix in range(n_items):
                        if rule == "generate_parse_strict":
                            is_correct = item_ix in (0, 2)
                        else:
                            is_correct = item_ix in (0, 1, 2)
                        if rule == "generate_parse_strict":
                            parseable = item_ix in (0, 2)
                        else:
                            parseable = item_ix != 3
                        parse_status = "parsed" if parseable else "unparseable"
                        rows.append({
                            "condition_id": cond_id,
                            "tier": "secondary",
                            "benchmark_id": "gsm8k",
                            "subject_id": None,
                            "model_id": model,
                            "model_family": "test_family",
                            "model_type": "base",
                            "parameter_count_b": 0.1,
                            "prompt_id": prompt,
                            "prompt_source_type": "benchmark_authors",
                            "seed_id": seed,
                            "fewshot_k": 5,
                            "scoring_rule_id": rule,
                            "task_type": "open_generation",
                            "item_id": f"gsm8k-{item_ix:03d}",
                            "gold_answer": "42",
                            "predicted_answer": "42" if is_correct else "0",
                            "is_correct": is_correct,
                            "parse_status": parse_status,
                            "parseable": parseable,
                            "choice_count": None,
                            "prompt_text_hash": sha256_str(f"GSMP/{prompt}"),
                            "rendered_prompt_hash": sha256_str(
                                f"GSMR/{model}/{rule}/{seed}/{item_ix}"
                            ),
                            "run_id": cond_id,
                            "selected_answer": None,
                            "ll_margin_top2": None,
                            "gold_choice_score": None,
                            "selected_choice_score": None,
                            "raw_generation": "The answer is 42." if is_correct else "Hmm, 0.",
                            "parsed_answer": "42" if is_correct else "0",
                            "generation_length_tokens": 6,
                        })
    return rows


def write_gsm8k_fixture_parquet(path: Path) -> None:
    """Write make_gsm8k_fixture_item_level() to a parquet file."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = make_gsm8k_fixture_item_level()
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def make_gsm8k_fixture_condition_level() -> list[dict]:
    """Aggregate the GSM8K item-level fixture to condition-level rows.

    One row per condition_id; mirrors what 05_make_condition_scores.py
    would produce on the same item-level fixture.
    """
    items = make_gsm8k_fixture_item_level()
    grouped: dict[str, list[dict]] = {}
    for r in items:
        grouped.setdefault(r["condition_id"], []).append(r)

    rows: list[dict] = []
    for cond_id in sorted(grouped):
        group = grouped[cond_id]
        first = group[0]
        n_items = len(group)
        n_correct = sum(1 for r in group if r["is_correct"])
        n_parseable = sum(1 for r in group if r.get("parseable"))
        rows.append({
            "condition_id": cond_id,
            "tier": first["tier"],
            "benchmark_id": first["benchmark_id"],
            "subject_id": first.get("subject_id") or "",
            "model_id": first["model_id"],
            "prompt_id": first["prompt_id"],
            "seed_id": first["seed_id"],
            "fewshot_k": first["fewshot_k"],
            "scoring_rule_id": first["scoring_rule_id"],
            "num_items": n_items,
            "num_correct": n_correct,
            "accuracy": n_correct / n_items if n_items else 0.0,
            "parseability_rate": n_parseable / n_items if n_items else 0.0,
            "mean_generation_length_tokens": 6.0,
            "condition_status": "complete",
        })
    return rows


def write_gsm8k_fixture_condition_csv(path: Path) -> None:
    rows = make_gsm8k_fixture_condition_level()
    fieldnames = list(rows[0].keys())
    write_csv(path, rows, fieldnames)


# --- Raw-run directory fixture ----------------------------------------------


def make_fixture_raw_run_dir(target_dir: Path, scoring_rule: str) -> Path:
    """Create a raw-run directory with metadata + the appropriate jsonl file.

    Used by tests/test_normalization.py to set up an end-to-end paper-root
    layout for `04_normalize_outputs.py`. The run directory name equals the
    canonical condition_id used elsewhere in this fixture file, so the
    metadata's condition_id and run_id agree by construction.

    Args:
        target_dir: Parent directory under which the run subdir is created.
        scoring_rule: Either "generate_parse" (G&P) or "ll_norm" (LL).

    Returns:
        Path to the created run directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    if scoring_rule == "generate_parse":
        metadata = make_tiny_run_metadata_gp()
        run_id = metadata["condition_id"]
        run_dir = target_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_metadata.json", metadata)
        write_jsonl(run_dir / "item_outputs.jsonl", make_tiny_item_outputs())
    elif scoring_rule == "ll_norm":
        metadata = make_tiny_run_metadata_ll()
        run_id = metadata["condition_id"]
        run_dir = target_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_metadata.json", metadata)
        write_jsonl(run_dir / "item_scores.jsonl", make_tiny_item_scores())
    else:
        raise ValueError(
            f"Unknown scoring_rule: {scoring_rule!r}. "
            f"Expected 'generate_parse' or 'll_norm'."
        )
    return run_dir


# --- Main -------------------------------------------------------------------


def main() -> int:
    # Tiny valid fixtures
    write_jsonl(FIXTURES_DIR / "tiny_item_outputs.jsonl", make_tiny_item_outputs())
    write_jsonl(FIXTURES_DIR / "tiny_item_scores.jsonl", make_tiny_item_scores())
    write_json(FIXTURES_DIR / "tiny_run_metadata_gp.json", make_tiny_run_metadata_gp())
    write_json(FIXTURES_DIR / "tiny_run_metadata_ll.json", make_tiny_run_metadata_ll())

    cond_rows = make_tiny_condition_scores()
    cond_fieldnames = list(cond_rows[0].keys())
    write_csv(FIXTURES_DIR / "tiny_condition_scores.csv", cond_rows, cond_fieldnames)

    # Pathological fixtures
    write_jsonl(FIXTURES_DIR / "pathological_all_parse_fail.jsonl",
                make_pathological_all_parse_fail())
    write_jsonl(FIXTURES_DIR / "pathological_near_tied_ll.jsonl",
                make_pathological_near_tied_ll())

    # Minimal config fixtures
    make_minimal_config_files(FIXTURES_DIR / "configs")

    # MMLU subject-decomp fixture (item-level parquet).
    write_mmlu_fixture_parquet(FIXTURES_DIR / "mmlu_item_level.parquet")

    # GSM8K case-study fixtures (item-level parquet + condition-level CSV).
    write_gsm8k_fixture_parquet(FIXTURES_DIR / "gsm8k_item_level.parquet")
    write_gsm8k_fixture_condition_csv(FIXTURES_DIR / "gsm8k_condition_level.csv")

    # Raw-run directory fixtures (one per scoring rule).
    raw_root = FIXTURES_DIR / "raw"
    if raw_root.exists():
        # Idempotent rebuild: clear stale per-run subdirs.
        import shutil
        try:
            shutil.rmtree(raw_root)
        except (PermissionError, OSError):
            # Already in place from a previous build and the filesystem
            # is read-only; tolerate and rebuild on top.
            pass
    make_fixture_raw_run_dir(raw_root, "generate_parse")
    make_fixture_raw_run_dir(raw_root, "ll_norm")

    # Mock analysis outputs (used by tests for 98_/99_ reporting scripts).
    mock_analysis_root = FIXTURES_DIR / "mock_analysis"
    if mock_analysis_root.exists():
        import shutil
        try:
            shutil.rmtree(mock_analysis_root)
        except (PermissionError, OSError):
            pass
    write_mock_analysis_outputs(mock_analysis_root)

    print(f"Fixtures written to {FIXTURES_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
