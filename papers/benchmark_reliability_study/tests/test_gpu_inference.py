"""Tests for scripts/gpu_inference.py.

Three layers of coverage:

1. Pure functions (parsing, prompt rendering, per-benchmark adapters,
   manifest loading, seed canonicalization). Run anywhere with stdlib
   only (plus jsonschema, pyyaml).

2. Atomic-write contract: builds a synthetic condition + rows and
   verifies _write_atomic produces schema-conformant output and never
   leaves a partial final directory.

3. Scoring functions (LL + G&P) using a tiny torch model with
   handcrafted logits. Skipped via pytest.importorskip when torch
   isn't installed (e.g. on local CPU machines without the pod env).

The tests are deliberately self-contained — they construct condition
rows, items, and manifests in-memory rather than depending on the
locked manifests in repo. The locked manifests are exercised by the
pod-side preflight, not by these unit tests.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_ROOT))


def _load_gpu_module():
    """Import scripts/gpu_inference.py as a module.

    Registers the module in sys.modules before exec — without this,
    dataclasses.KW_ONLY introspection of ``str | None`` annotations
    fails because cls.__module__ resolves to None during decoration.
    """
    spec = importlib.util.spec_from_file_location(
        "gpu_inference",
        PAPER_ROOT / "scripts" / "gpu_inference.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GPU = _load_gpu_module()


# ---------------------------------------------------------------------------
# 1. Pure-function tests
# ---------------------------------------------------------------------------


class TestNormalizeSeed:
    def test_strips_s_prefix(self):
        assert GPU._normalize_seed("s42") == "42"
        assert GPU._normalize_seed("s2024") == "2024"

    def test_passes_bare_int_through(self):
        assert GPU._normalize_seed("42") == "42"

    def test_zero_shot_marker(self):
        assert GPU._normalize_seed("0shot") is None

    def test_blank_or_none(self):
        assert GPU._normalize_seed("") is None
        assert GPU._normalize_seed(None) is None

    def test_does_not_strip_non_numeric_after_s(self):
        # "ssomething" should not be misread as a seed prefix
        assert GPU._normalize_seed("sfoo") == "sfoo"


class TestRenderChoicesBlock:
    def test_letters_assigned_in_order(self):
        block = GPU.render_choices_block(["alpha", "beta", "gamma"])
        assert block == "A) alpha\nB) beta\nC) gamma"

    def test_two_choice(self):
        assert GPU.render_choices_block(["yes", "no"]) == "A) yes\nB) no"


class TestParseAnswer:
    def test_empty_generation(self):
        status, parsed = GPU.parse_answer("", "benchmark_specific_regex", 4)
        assert status == "empty"
        assert parsed is None

    def test_whitespace_only_is_empty(self):
        status, _ = GPU.parse_answer("   \n  ", "benchmark_specific_regex", 4)
        assert status == "empty"

    def test_bsr_first_letter(self):
        status, parsed = GPU.parse_answer(" B", "benchmark_specific_regex", 4)
        assert status == "parsed"
        assert parsed == "B"

    def test_bsr_letter_outside_choice_space(self):
        # 4-choice question, model says "E"
        status, parsed = GPU.parse_answer(" E", "benchmark_specific_regex", 4)
        assert status == "invalid_choice"
        assert parsed == "E"

    def test_bsr_no_letter(self):
        status, _ = GPU.parse_answer(" the answer is 42", "benchmark_specific_regex", 4)
        assert status == "unparseable"

    def test_bsr_multiple_distinct_letters_early(self):
        status, _ = GPU.parse_answer(" A or B", "benchmark_specific_regex", 4)
        assert status == "multiple_answers"

    def test_bsr_repeated_letter_is_parsed(self):
        # "A. A is correct" — same letter, not multiple_answers
        status, parsed = GPU.parse_answer(" A. A is correct.", "benchmark_specific_regex", 4)
        assert status == "parsed"
        assert parsed == "A"

    def test_gsm_strict_simple(self):
        status, parsed = GPU.parse_answer(" The answer is 42", "exact_match_final_number", None)
        assert status == "parsed"
        assert parsed == "42"

    def test_gsm_strict_with_decimal(self):
        status, parsed = GPU.parse_answer("3.14", "exact_match_final_number", None)
        assert status == "parsed"
        assert parsed == "3.14"

    def test_gsm_strict_no_trailing_number(self):
        status, _ = GPU.parse_answer("the cat sat", "exact_match_final_number", None)
        assert status == "unparseable"

    def test_gsm_permissive_with_commas(self):
        status, parsed = GPU.parse_answer(
            "Final amount: 1,250.50", "permissive_number_regex", None,
        )
        assert status == "parsed"
        assert parsed == "1250.50"

    def test_gsm_permissive_takes_last_number(self):
        # The permissive parser searches the last 5 lines, last match.
        status, parsed = GPU.parse_answer(
            "Step 1: 10 apples\nStep 2: 5 each\nTotal: 50",
            "permissive_number_regex", None,
        )
        assert status == "parsed"
        assert parsed == "50"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            GPU.parse_answer("x", "no_such_strategy", 4)


class TestPerBenchmarkAdapters:
    def test_arc_question_choices_gold(self):
        item = {
            "question": "What color is the sky?",
            "choices": {"text": ["red", "blue", "green", "yellow"],
                        "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
            "id": "Mercury_123",
        }
        assert GPU.get_question(item, "arc_challenge") == "What color is the sky?"
        assert GPU.get_choices(item, "arc_challenge") == ["red", "blue", "green", "yellow"]
        assert GPU.get_gold_letter(item, "arc_challenge") == "B"

    def test_arc_with_numeric_labels(self):
        # Some ARC items use 1-4 instead of A-D
        item = {
            "question": "Q?",
            "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
            "answerKey": "3",
            "id": "x1",
        }
        # "3" is at index 2 → letter "C"
        assert GPU.get_gold_letter(item, "arc_challenge") == "C"

    def test_hellaswag_question_concatenates_ctx(self):
        item = {"ctx_a": "He climbs the ladder.", "ctx_b": "Then he", "endings": ["a", "b", "c", "d"], "label": "1"}
        q = GPU.get_question(item, "hellaswag")
        assert "ladder" in q and "Then he" in q

    def test_hellaswag_gold_from_label_string(self):
        item = {"label": "2", "endings": ["a", "b", "c", "d"], "ctx_a": "x", "ctx_b": ""}
        assert GPU.get_gold_letter(item, "hellaswag") == "C"

    def test_truthfulqa_mc_choices_and_gold(self):
        item = {
            "question": "Q?",
            "mc1_targets": {
                "choices": ["Right", "Wrong1", "Wrong2"],
                "labels": [1, 0, 0],
            },
        }
        assert GPU.get_choices(item, "truthfulqa_mc") == ["Right", "Wrong1", "Wrong2"]
        assert GPU.get_gold_letter(item, "truthfulqa_mc") == "A"

    def test_truthfulqa_mc_variable_choice_count(self):
        item = {
            "question": "Q?",
            "mc1_targets": {
                "choices": ["a", "b", "c", "d", "e"],
                "labels": [0, 0, 0, 1, 0],
            },
        }
        assert GPU.get_gold_letter(item, "truthfulqa_mc") == "D"

    def test_mmlu_subject_and_gold(self):
        item = {
            "question": "Q?", "choices": ["w", "x", "y", "z"],
            "answer": 2, "subject": "world_religions",
        }
        assert GPU.get_choices(item, "mmlu_panel") == ["w", "x", "y", "z"]
        assert GPU.get_gold_letter(item, "mmlu_panel") == "C"
        assert GPU.get_subject(item, "mmlu_panel") == "world_religions"

    def test_winogrande_two_choice(self):
        item = {"sentence": "The trophy doesn't fit in the suitcase because _ is too large.",
                "option1": "trophy", "option2": "suitcase", "answer": "1"}
        assert GPU.get_choices(item, "winogrande") == ["trophy", "suitcase"]
        assert GPU.get_gold_letter(item, "winogrande") == "A"
        assert GPU.get_question(item, "winogrande") == item["sentence"]

    def test_gsm8k_gold_text(self):
        item = {"question": "What is 2+2?", "answer": "Adding gives 4.\n#### 4"}
        assert GPU.get_gold_text(item, "gsm8k") == "4"

    def test_gsm8k_gold_text_with_commas(self):
        item = {"answer": "blah blah\n#### 1,234"}
        assert GPU.get_gold_text(item, "gsm8k") == "1234"

    def test_gsm8k_get_choices_raises(self):
        with pytest.raises(ValueError):
            GPU.get_choices({}, "gsm8k")

    def test_gsm8k_get_gold_letter_raises(self):
        with pytest.raises(ValueError):
            GPU.get_gold_letter({}, "gsm8k")

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            GPU.get_choices({}, "no_such_benchmark")


class TestRenderFewshotExample:
    def test_constrained_choice_format(self):
        item = {
            "question": "Q?",
            "choices": {"text": ["a", "b"], "label": ["A", "B"]},
            "answerKey": "B",
        }
        # Use ARC adapter
        block = GPU.render_fewshot_example(item, "arc_challenge")
        assert "Question: Q?" in block
        assert "A) a" in block and "B) b" in block
        assert block.endswith("Answer: B\n\n")

    def test_gsm8k_includes_full_answer_chain(self):
        item = {"question": "What is 2+2?", "answer": "Reasoning…\n#### 4"}
        block = GPU.render_fewshot_example(item, "gsm8k")
        assert "Reasoning" in block
        assert "#### 4" in block
        assert block.endswith("\n\n")


class TestRenderPrompt:
    @pytest.fixture
    def arc_benchmark(self):
        from gradience_study.config import BenchmarkSpec
        return BenchmarkSpec(
            benchmark_id="arc_challenge", display_name="ARC", tier="primary",
            hf_dataset="allenai/ai2_arc", hf_config="ARC-Challenge",
            dataset_version_hash="x", task_type="constrained_choice",
            fewshot_k=0, seed_facet=False, fewshot_source_split=None,
            eval_split="test", scoring_rules=("ll_norm",),
            item_id_field="id", answer_field="answerKey", expected_num_items=10,
        )

    @pytest.fixture
    def prompt_spec(self):
        from gradience_study.config import PromptSpec
        return PromptSpec(
            benchmark_id="arc_challenge", prompt_id="P_test",
            prompt_path="prompts/x", content_hash="x",
            source_type="author_minimal_declared", source_url_or_ref="x",
            source_commit=None, admissibility_status="locked",
            placeholders_used=("fewshot_examples", "question", "choices"),
        )

    def test_zero_shot_substitution(self, arc_benchmark, prompt_spec):
        template = "{{fewshot_examples}}Question: {{question}}\n{{choices}}\nAnswer:"
        item = {"question": "Q?", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
                "answerKey": "A", "id": "x"}
        rendered = GPU.render_prompt(template, item, [], arc_benchmark, prompt_spec)
        assert "{{" not in rendered
        assert rendered.startswith("Question: Q?")
        assert "A) a" in rendered

    def test_unfilled_placeholder_raises(self, arc_benchmark, prompt_spec):
        template = "{{fewshot_examples}}Q: {{undefined_placeholder}}"
        item = {"question": "Q?", "choices": {"text": ["a"], "label": ["A"]},
                "answerKey": "A", "id": "x"}
        with pytest.raises(RuntimeError, match="Unfilled placeholder"):
            GPU.render_prompt(template, item, [], arc_benchmark, prompt_spec)

    def test_fewshot_block_concatenates(self, arc_benchmark, prompt_spec):
        template = "{{fewshot_examples}}Question: {{question}}"
        fs_items = [
            {"question": "F1", "choices": {"text": ["a"], "label": ["A"]}, "answerKey": "A", "id": "f1"},
            {"question": "F2", "choices": {"text": ["b"], "label": ["A"]}, "answerKey": "A", "id": "f2"},
        ]
        item = {"question": "Real", "choices": {"text": ["c"], "label": ["A"]}, "answerKey": "A", "id": "x"}
        rendered = GPU.render_prompt(template, item, fs_items, arc_benchmark, prompt_spec)
        assert "F1" in rendered and "F2" in rendered
        # 2 fewshot blocks + 1 final question, all formatted as "Question: "
        assert rendered.count("Question: ") == 3
        assert rendered.endswith("Question: Real")


# ---------------------------------------------------------------------------
# 2. Manifest loading + atomic-write contract
# ---------------------------------------------------------------------------


class TestLoadManifests:
    def test_load_conditions_round_trip(self, tmp_path):
        path = tmp_path / "conditions.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "condition_id", "tier", "benchmark_id", "subject_id",
                "model_id", "prompt_id", "seed_id", "fewshot_k",
                "scoring_rule_id", "task_type", "expected_num_items",
                "condition_status", "created_at", "config_hash",
            ])
            w.writerow([
                "arc__pythia_410m__P1__s42__ll_norm", "primary", "arc_challenge",
                "", "pythia_410m", "P1", "s42", "5",
                "ll_norm", "constrained_choice", "1172", "pending",
                "2026-04-25T00:00:00Z", "abc12345",
            ])
        conds = GPU.load_conditions(path)
        assert len(conds) == 1
        assert conds[0].subject_id is None
        assert conds[0].seed_id == "s42"
        assert conds[0].fewshot_k == 5

    def test_load_fewshot_manifest_canonicalizes_seed(self, tmp_path):
        path = tmp_path / "fewshot.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "benchmark_id", "subject_id", "seed_id", "fewshot_k",
                "example_rank", "example_item_id", "example_source_split",
                "draw_timestamp", "draw_script_version",
            ])
            for rank, iid in enumerate(["a", "b", "c"]):
                w.writerow([
                    "arc_challenge", "", "42", "3",
                    str(rank), f"train:{iid}", "train",
                    "2026-04-25T00:00:00Z", "0.1.0",
                ])
        lookup = GPU.load_fewshot_manifest(path)
        # Lookup with canonical seed "42" works
        assert ("arc_challenge", None, "42") in lookup
        assert lookup[("arc_challenge", None, "42")] == [
            "train:a", "train:b", "train:c",
        ]


class TestAtomicWrite:
    """Smoke test the .tmp/ → final rename contract.

    Builds synthetic rows that match the schemas, calls _write_atomic,
    and verifies the final dir contains run_metadata.json + the right
    JSONL file. Then asserts a second call with the same condition_id
    raises rather than silently overwriting.
    """

    def _build_inputs(self, tmp_path):
        from gradience_study.config import (
            BenchmarkSpec, ModelSpec, PromptSpec, ScoringRuleSpec,
        )
        condition = GPU.ConditionRow(
            condition_id="arc__pythia_410m__P1__s42__ll_norm",
            tier="primary", benchmark_id="arc_challenge", subject_id=None,
            model_id="pythia_410m", prompt_id="P1", seed_id="s42",
            fewshot_k=5, scoring_rule_id="ll_norm",
            task_type="constrained_choice", expected_num_items=2,
            condition_status="pending", config_hash="abc12345",
        )
        benchmark = BenchmarkSpec(
            benchmark_id="arc_challenge", display_name="ARC", tier="primary",
            hf_dataset="allenai/ai2_arc", hf_config="ARC-Challenge",
            dataset_version_hash="x", task_type="constrained_choice",
            fewshot_k=5, seed_facet=True, fewshot_source_split="train",
            eval_split="test", scoring_rules=("ll_norm",),
            item_id_field="id", answer_field="answerKey", expected_num_items=2,
        )
        prompt_spec = PromptSpec(
            benchmark_id="arc_challenge", prompt_id="P1",
            prompt_path="prompts/x", content_hash="x" * 64,
            source_type="benchmark_authors", source_url_or_ref="x",
            source_commit=None, admissibility_status="locked",
            placeholders_used=("fewshot_examples", "question", "choices"),
        )
        scoring_rule = ScoringRuleSpec(
            scoring_rule_id="ll_norm",
            display_name="Length-Normalized Log-Likelihood",
            applies_to_task_types=("constrained_choice",),
            normalization="length_normalized", selection_rule="argmax",
        )
        ctx = GPU.ConditionContext(
            benchmark=benchmark, prompt_spec=prompt_spec, scoring_rule=scoring_rule,
            template_text="x", template_hash="0" * 64,
            eval_items=[], fewshot_items=[], fewshot_k=5,
        )
        model_spec = ModelSpec(
            model_id="pythia_410m", hf_name="EleutherAI/pythia-410m",
            hf_revision="9879c9b5",
            model_family="pythia", model_type="base",
            parameter_count_b=0.41, primary=True, optional_extension=False,
        )
        rows = [
            {
                "condition_id": condition.condition_id, "item_id": "x1",
                "subject_id": None,
                "prompt_text_hash": "a" * 64, "rendered_prompt_hash": "b" * 64,
                "scoring_rule_id": "ll_norm", "gold_answer": "A",
                "choices": ["A", "B"], "choice_scores": {"A": -1.0, "B": -2.0},
                "choice_token_counts": {"A": 1, "B": 1},
                "normalization": "length_normalized", "selected_answer": "A",
                "is_correct": True,
            },
            {
                "condition_id": condition.condition_id, "item_id": "x2",
                "subject_id": None,
                "prompt_text_hash": "a" * 64, "rendered_prompt_hash": "c" * 64,
                "scoring_rule_id": "ll_norm", "gold_answer": "B",
                "choices": ["A", "B"], "choice_scores": {"A": -2.0, "B": -1.0},
                "choice_token_counts": {"A": 1, "B": 1},
                "normalization": "length_normalized", "selected_answer": "B",
                "is_correct": True,
            },
        ]
        schemas = GPU._load_schemas(PAPER_ROOT / "schemas")
        return condition, ctx, rows, schemas, model_spec

    def test_atomic_write_produces_valid_dir(self, tmp_path):
        condition, ctx, rows, schemas, model_spec = self._build_inputs(tmp_path)
        out_dir = tmp_path / "raw"
        GPU._write_atomic(
            out_dir, condition, ctx, rows,
            started_at=1.0, finished_at=2.0,
            schemas=schemas, model_spec=model_spec,
            transformers_version="4.46.0", torch_version="2.4.1",
        )
        final = out_dir / condition.condition_id
        assert final.is_dir()
        assert (final / "run_metadata.json").exists()
        assert (final / "item_scores.jsonl").exists()
        meta = json.loads((final / "run_metadata.json").read_text())
        assert meta["status"] == "complete"
        assert meta["num_items_completed"] == 2
        assert meta["run_id"] == meta["condition_id"]
        with open(final / "item_scores.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert all(line["scoring_rule_id"] == "ll_norm" for line in lines)
        # Tmp dir must be cleaned up by the rename
        assert not (out_dir / ".tmp" / condition.condition_id).exists()

    def test_double_write_refuses(self, tmp_path):
        condition, ctx, rows, schemas, model_spec = self._build_inputs(tmp_path)
        out_dir = tmp_path / "raw"
        GPU._write_atomic(
            out_dir, condition, ctx, rows, 1.0, 2.0,
            schemas, model_spec, "4.46.0", "2.4.1",
        )
        with pytest.raises(RuntimeError, match="Final dir already exists"):
            GPU._write_atomic(
                out_dir, condition, ctx, rows, 3.0, 4.0,
                schemas, model_spec, "4.46.0", "2.4.1",
            )


class TestFilterPending:
    def test_skips_completed_conditions(self, tmp_path):
        cond_a = GPU.ConditionRow(
            condition_id="c_a", tier="primary", benchmark_id="b", subject_id=None,
            model_id="m", prompt_id="P", seed_id="s42", fewshot_k=5,
            scoring_rule_id="ll_norm", task_type="constrained_choice",
            expected_num_items=1, condition_status="pending",
            config_hash="x",
        )
        cond_b = GPU.ConditionRow(
            condition_id="c_b", tier="primary", benchmark_id="b", subject_id=None,
            model_id="m", prompt_id="P", seed_id="s42", fewshot_k=5,
            scoring_rule_id="ll_norm", task_type="constrained_choice",
            expected_num_items=1, condition_status="pending",
            config_hash="x",
        )
        out_dir = tmp_path / "raw"
        # Pretend cond_a already ran
        (out_dir / "c_a").mkdir(parents=True)
        (out_dir / "c_a" / "run_metadata.json").write_text("{}")
        pending = GPU._filter_pending([cond_a, cond_b], out_dir)
        assert [c.condition_id for c in pending] == ["c_b"]


class TestDropUnderscoreKeys:
    def test_strips_underscore_prefixed(self):
        out = GPU._drop_underscore_keys({"a": 1, "_b": 2, "c_": 3})
        assert out == {"a": 1, "c_": 3}


# ---------------------------------------------------------------------------
# 3. Scoring functions (skipped without torch)
# ---------------------------------------------------------------------------


class TestScoring:
    """Integration tests for LL/G&P scoring with synthetic logits.

    Uses a hand-built tiny model that returns deterministic logits so
    we can verify the math of length-normalized LL and the dispatch
    of G&P parsing.
    """

    @pytest.fixture(scope="class")
    def torch_module(self):
        return pytest.importorskip("torch")

    def test_score_ll_norm_picks_argmax(self, torch_module):
        """A model that always assigns highest logit to ' A' must select A."""
        torch = torch_module

        class FakeTok:
            # Tokens 0='<pad>', 1=' A', 2=' B', 3=' C', 4=' D', 5='X' (prompt)
            vocab_size = 6
            pad_token_id = 0
            eos_token_id = 0

            def encode(self, text, add_special_tokens=True, return_tensors=None):
                if text == "prompt":
                    ids = [5]
                else:
                    # text is " A" / " B" / " C" / " D"
                    letter_to_id = {" A": 1, " B": 2, " C": 3, " D": 4}
                    ids = [letter_to_id[text]]
                if return_tensors == "pt":
                    return torch.tensor([ids], dtype=torch.long)
                return ids

        class FakeOut:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel:
            def __init__(self):
                self.device = torch.device("cpu")

            def __call__(self, input_ids):
                # For each position, return logits that strongly prefer token 1 (=" A")
                B, T = input_ids.shape
                logits = torch.full((B, T, 6), -10.0)
                logits[:, :, 1] = 10.0  # token 1 (" A") gets the boost
                return FakeOut(logits)

        out = GPU.score_ll_norm(FakeModel(), FakeTok(), "prompt", n_choices=4)
        assert out["selected_answer"] == "A"
        # All choice scores are length-normalized log-probs; A is highest
        assert out["choice_scores"]["A"] > out["choice_scores"]["B"]
        assert set(out["choices"]) == {"A", "B", "C", "D"}
        assert out["normalization"] == "length_normalized"

    def test_score_generate_parse_dispatches_strategy(self, torch_module):
        """A G&P generation that produces ' B\\n' must parse to letter 'B'."""
        torch = torch_module

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, return_tensors=None):
                ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
                attn = torch.ones_like(ids)

                class _BatchEncoding:
                    def __init__(self, ids, attn):
                        self.input_ids = ids
                        self.attention_mask = attn

                    def to(self, device):
                        return self  # already on cpu in this test

                    def get(self, key):
                        return getattr(self, key, None)

                return _BatchEncoding(ids, attn)

            def decode(self, ids, skip_special_tokens=True):
                return " B explanation"

        class FakeModel:
            def __init__(self):
                self.device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, max_new_tokens,
                         do_sample, num_beams, pad_token_id):
                # Append 5 fake new tokens
                new = torch.zeros((1, 5), dtype=torch.long)
                return torch.cat([input_ids, new], dim=1)

        out = GPU.score_generate_parse(
            FakeModel(), FakeTok(), prompt="x",
            parse_strategy="benchmark_specific_regex",
            n_choices=4, max_new_tokens=32,
        )
        assert out["parse_status"] == "parsed"
        assert out["parsed_answer"] == "B"
        assert out["raw_generation"] == " B explanation"
        assert out["generation_length_tokens"] == 5


# ---------------------------------------------------------------------------
# 4. Schema-conformance smoke (locked schemas vs. tiny fixtures)
# ---------------------------------------------------------------------------


class TestSchemaConformance:
    """Verify the locked tiny fixtures (committed by prior session) still
    validate against schemas exactly as gpu_inference would emit. This
    pins the GPU/CPU contract from both ends."""

    def test_tiny_run_metadata_ll_validates(self):
        jsonschema = pytest.importorskip("jsonschema")
        schemas = GPU._load_schemas(PAPER_ROOT / "schemas")
        meta = json.loads((PAPER_ROOT / "tests/fixtures/tiny_run_metadata_ll.json").read_text())
        jsonschema.validate(meta, schemas["run_metadata"])

    def test_tiny_run_metadata_gp_validates(self):
        jsonschema = pytest.importorskip("jsonschema")
        schemas = GPU._load_schemas(PAPER_ROOT / "schemas")
        meta = json.loads((PAPER_ROOT / "tests/fixtures/tiny_run_metadata_gp.json").read_text())
        jsonschema.validate(meta, schemas["run_metadata"])

    def test_tiny_item_scores_validate(self):
        jsonschema = pytest.importorskip("jsonschema")
        schemas = GPU._load_schemas(PAPER_ROOT / "schemas")
        with open(PAPER_ROOT / "tests/fixtures/tiny_item_scores.jsonl") as f:
            for line in f:
                if line.strip():
                    jsonschema.validate(json.loads(line), schemas["item_scores"])

    def test_tiny_item_outputs_validate(self):
        jsonschema = pytest.importorskip("jsonschema")
        schemas = GPU._load_schemas(PAPER_ROOT / "schemas")
        with open(PAPER_ROOT / "tests/fixtures/tiny_item_outputs.jsonl") as f:
            for line in f:
                if line.strip():
                    jsonschema.validate(json.loads(line), schemas["item_outputs"])
