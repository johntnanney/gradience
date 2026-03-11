"""Tests for gradience.vnext.merge.eligibility — source adapter screening."""

from __future__ import annotations

from gradience.vnext.merge.eligibility import (
    AdapterQAResult,
    EligibilityStatus,
    classify_eligibility,
    screen_adapters,
)

# ---------------------------------------------------------------------------
# EligibilityStatus
# ---------------------------------------------------------------------------


class TestEligibilityStatus:
    def test_values(self):
        assert EligibilityStatus.ELIGIBLE.value == "eligible"
        assert EligibilityStatus.UNCERTAIN.value == "uncertain"
        assert EligibilityStatus.FLAGGED_WEAK.value == "flagged_weak"
        assert EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL.value == "unknown_no_behavioral_eval"

    def test_is_str_enum(self):
        assert isinstance(EligibilityStatus.ELIGIBLE, str)
        assert EligibilityStatus.ELIGIBLE == "eligible"


# ---------------------------------------------------------------------------
# AdapterQAResult
# ---------------------------------------------------------------------------


class TestAdapterQAResult:
    def test_construction_defaults(self):
        qa = AdapterQAResult(
            adapter_path="/path/to/adapter",
            status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
        )
        assert qa.adapter_path == "/path/to/adapter"
        assert qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        assert qa.adapter_metric is None
        assert qa.base_metric is None
        assert qa.metric_name == ""
        assert qa.lower_is_better is True
        assert qa.evidence == {}

    def test_to_dict_roundtrip(self):
        qa = AdapterQAResult(
            adapter_path="./adapter_a",
            status=EligibilityStatus.FLAGGED_WEAK,
            adapter_metric=5.47,
            base_metric=4.66,
            metric_name="perplexity",
            lower_is_better=True,
            eval_dataset="oasst2",
            notes="Weaker than base.",
            evidence={"source": "study16"},
        )
        d = qa.to_dict()
        restored = AdapterQAResult.from_dict(d)
        assert restored.adapter_path == qa.adapter_path
        assert restored.status == qa.status
        assert restored.adapter_metric == qa.adapter_metric
        assert restored.base_metric == qa.base_metric
        assert restored.metric_name == qa.metric_name
        assert restored.lower_is_better == qa.lower_is_better
        assert restored.eval_dataset == qa.eval_dataset
        assert restored.notes == qa.notes

    def test_from_dict_unknown_status(self):
        qa = AdapterQAResult.from_dict({"status": "bogus_value"})
        assert qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

    def test_from_dict_empty(self):
        qa = AdapterQAResult.from_dict({})
        assert qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        assert qa.adapter_path == ""


# ---------------------------------------------------------------------------
# classify_eligibility
# ---------------------------------------------------------------------------


class TestClassifyEligibility:
    def test_no_metrics_returns_unknown(self):
        qa = classify_eligibility("./adapter", None, None)
        assert qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

    def test_adapter_metric_only_returns_unknown(self):
        qa = classify_eligibility("./adapter", 2.5, None)
        assert qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

    def test_lower_is_better_eligible(self):
        # Adapter PPL 2.11, base PPL 2.86 → adapter beats base
        qa = classify_eligibility(
            "./metamath",
            adapter_metric=2.11,
            base_metric=2.86,
            metric_name="perplexity",
            lower_is_better=True,
        )
        assert qa.status == EligibilityStatus.ELIGIBLE
        assert "beats base" in qa.notes

    def test_lower_is_better_weak(self):
        # Adapter PPL 5.47, base PPL 4.66 → adapter is worse
        qa = classify_eligibility(
            "./btgenbot",
            adapter_metric=5.47,
            base_metric=4.66,
            metric_name="perplexity",
            lower_is_better=True,
        )
        assert qa.status == EligibilityStatus.FLAGGED_WEAK
        assert "worse than base" in qa.notes

    def test_lower_is_better_uncertain(self):
        # Adapter PPL 2.85, base PPL 2.75, margin=0.2 → within margin
        qa = classify_eligibility(
            "./magicoder",
            adapter_metric=2.85,
            base_metric=2.75,
            metric_name="perplexity",
            lower_is_better=True,
            margin=0.2,
        )
        # delta = 2.75 - 2.85 = -0.10, within [-0.2, 0.2] → uncertain
        assert qa.status == EligibilityStatus.UNCERTAIN

    def test_lower_is_better_margin_uncertain(self):
        # Adapter just barely beats base but within margin
        qa = classify_eligibility(
            "./adapter",
            adapter_metric=2.80,
            base_metric=2.86,
            metric_name="perplexity",
            lower_is_better=True,
            margin=0.1,
        )
        # delta = 2.86 - 2.80 = 0.06, > 0 but < margin → uncertain
        assert qa.status == EligibilityStatus.UNCERTAIN

    def test_higher_is_better_eligible(self):
        qa = classify_eligibility(
            "./adapter",
            adapter_metric=0.85,
            base_metric=0.72,
            metric_name="accuracy",
            lower_is_better=False,
        )
        assert qa.status == EligibilityStatus.ELIGIBLE

    def test_higher_is_better_weak(self):
        qa = classify_eligibility(
            "./adapter",
            adapter_metric=0.65,
            base_metric=0.72,
            metric_name="accuracy",
            lower_is_better=False,
        )
        assert qa.status == EligibilityStatus.FLAGGED_WEAK

    def test_eval_dataset_preserved(self):
        qa = classify_eligibility(
            "./adapter",
            adapter_metric=2.11,
            base_metric=2.86,
            eval_dataset="gsm8k",
        )
        assert qa.eval_dataset == "gsm8k"

    def test_uncertain_symmetric_negative_delta_within_margin(self):
        """Small negative delta within margin should be uncertain, not flagged_weak."""
        result = classify_eligibility(
            adapter_path="./test",
            adapter_metric=4.70,  # slightly worse
            base_metric=4.66,  # base is better (lower is better)
            metric_name="perplexity",
            lower_is_better=True,
            margin=0.1,
        )
        # delta = 4.66 - 4.70 = -0.04, within [-0.1, 0.1] → uncertain
        assert result.status == EligibilityStatus.UNCERTAIN


# ---------------------------------------------------------------------------
# screen_adapters
# ---------------------------------------------------------------------------


class TestScreenAdapters:
    def test_no_qa_no_warnings(self):
        warnings = screen_adapters(None, None)
        assert warnings == []

    def test_eligible_no_warnings(self):
        qa_a = AdapterQAResult(adapter_path="./a", status=EligibilityStatus.ELIGIBLE)
        qa_b = AdapterQAResult(adapter_path="./b", status=EligibilityStatus.ELIGIBLE)
        warnings = screen_adapters(qa_a, qa_b)
        assert warnings == []

    def test_one_weak_warns(self):
        qa_a = AdapterQAResult(adapter_path="./a", status=EligibilityStatus.ELIGIBLE)
        qa_b = AdapterQAResult(
            adapter_path="./btgenbot",
            status=EligibilityStatus.FLAGGED_WEAK,
            metric_name="perplexity",
            eval_dataset="oasst2",
        )
        warnings = screen_adapters(qa_a, qa_b)
        assert len(warnings) == 1
        assert "btgenbot" in warnings[0]
        assert "weaker" in warnings[0].lower()
        assert "perplexity" in warnings[0]
        assert "oasst2" in warnings[0]

    def test_both_weak_extra_warning(self):
        qa_a = AdapterQAResult(
            adapter_path="./catsubcat",
            status=EligibilityStatus.FLAGGED_WEAK,
        )
        qa_b = AdapterQAResult(
            adapter_path="./btgenbot",
            status=EligibilityStatus.FLAGGED_WEAK,
        )
        warnings = screen_adapters(qa_a, qa_b)
        # Individual warnings + the "both weak" warning
        assert len(warnings) == 3
        assert any("both" in w.lower() for w in warnings)
        assert any("ill-posed" in w.lower() for w in warnings)

    def test_unknown_warns_about_structural_only(self):
        qa_a = AdapterQAResult(adapter_path="./a", status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL)
        warnings = screen_adapters(qa_a, None)
        assert len(warnings) == 1
        assert "structural analysis only" in warnings[0].lower()

    def test_uncertain_warns(self):
        qa_a = AdapterQAResult(
            adapter_path="./a",
            status=EligibilityStatus.UNCERTAIN,
            metric_name="perplexity",
        )
        warnings = screen_adapters(qa_a, None)
        assert len(warnings) == 1
        assert "uncertain" in warnings[0].lower()
        assert "perplexity" in warnings[0]


# ---------------------------------------------------------------------------
# Import from merge __init__
# ---------------------------------------------------------------------------


class TestEligibilityContextProperties:
    def test_eligibility_context_any_unverified(self):
        """any_unverified should be True when either adapter has unknown status."""
        from gradience.vnext.merge.recommend import EligibilityContext

        ctx = EligibilityContext(
            status_a=EligibilityStatus.ELIGIBLE,
            status_b=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
        )
        assert ctx.any_unverified is True

    def test_eligibility_context_not_unverified_when_both_known(self):
        """any_unverified should be False when both have behavioral eval."""
        from gradience.vnext.merge.recommend import EligibilityContext

        ctx = EligibilityContext(
            status_a=EligibilityStatus.ELIGIBLE,
            status_b=EligibilityStatus.UNCERTAIN,
        )
        assert ctx.any_unverified is False


class TestPublicAPI:
    def test_importable_from_merge(self):
        from gradience.vnext.merge import (
            EligibilityStatus,
        )

        assert EligibilityStatus.ELIGIBLE.value == "eligible"
