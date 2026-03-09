"""
Tests for the escalation (safety fallback) module.

Covers: failure classification, candidate computation, orchestration,
verdict enrichment, and end-to-end integration.
"""

from __future__ import annotations

import unittest

from gradience.bench.escalation import (
    EscalationTrace,
    EscalationTraceEntry,
    classify_failure,
    compute_escalation_candidate,
    enrich_verdicts_with_stability,
    run_escalation_round,
    update_escalation_trace_with_results,
)

# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure(unittest.TestCase):
    """Tests for classify_failure()."""

    def test_pass_within_tolerance(self):
        """Delta inside tolerance → 'none'."""
        self.assertEqual(classify_failure(-0.003, acc_tolerance=0.005), "none")

    def test_positive_delta_is_none(self):
        """Positive delta (compressed is better) → 'none'."""
        self.assertEqual(classify_failure(+0.01, acc_tolerance=0.005), "none")

    def test_edge_at_tolerance_boundary(self):
        """Delta exactly at -tolerance → still 'none' (>= check)."""
        self.assertEqual(classify_failure(-0.005, acc_tolerance=0.005), "none")

    def test_near_miss(self):
        """Delta below tolerance but above catastrophic threshold → 'near_miss'.

        tolerance=0.005, margin=max(0.01, 0.5*0.005)=0.01
        catastrophic threshold = -(0.005 + 0.01) = -0.015
        delta=-0.010 is between -0.005 and -0.015 → near_miss
        """
        self.assertEqual(classify_failure(-0.010, acc_tolerance=0.005), "near_miss")

    def test_catastrophic(self):
        """Large negative delta → 'catastrophic'."""
        self.assertEqual(classify_failure(-0.060, acc_tolerance=0.005), "catastrophic")

    def test_catastrophic_with_large_tolerance(self):
        """tolerance=0.02, margin=max(0.01, 0.5*0.02)=0.01
        catastrophic threshold = -(0.02 + 0.01) = -0.03
        delta=-0.060 → catastrophic
        """
        self.assertEqual(classify_failure(-0.060, acc_tolerance=0.02), "catastrophic")

    def test_near_miss_with_large_tolerance(self):
        """tolerance=0.02, margin=0.01, threshold=-0.03
        delta=-0.025 is between -0.02 and -0.03 → near_miss
        """
        self.assertEqual(classify_failure(-0.025, acc_tolerance=0.02), "near_miss")

    def test_custom_margin(self):
        """Explicit catastrophic_margin overrides auto-computation.

        tolerance=0.005, margin=0.02, threshold=-(0.005+0.02)=-0.025
        delta=-0.020 → near_miss (between -0.005 and -0.025)
        """
        self.assertEqual(
            classify_failure(-0.020, acc_tolerance=0.005, catastrophic_margin=0.02),
            "near_miss",
        )

    def test_custom_margin_catastrophic(self):
        """Custom margin, delta beyond threshold → catastrophic."""
        self.assertEqual(
            classify_failure(-0.030, acc_tolerance=0.005, catastrophic_margin=0.02),
            "catastrophic",
        )

    def test_edge_at_catastrophic_boundary(self):
        """Delta exactly at catastrophic threshold → 'near_miss' (not <, so not catastrophic).

        tolerance=0.005, margin=0.01, threshold=-0.015
        delta=-0.015 is NOT < -0.015, so near_miss.
        """
        self.assertEqual(classify_failure(-0.015, acc_tolerance=0.005), "near_miss")

    def test_just_past_catastrophic_boundary(self):
        """Delta just below catastrophic threshold → 'catastrophic'."""
        self.assertEqual(classify_failure(-0.0151, acc_tolerance=0.005), "catastrophic")


# ---------------------------------------------------------------------------
# compute_escalation_candidate
# ---------------------------------------------------------------------------


class TestComputeEscalationCandidate(unittest.TestCase):
    """Tests for compute_escalation_candidate()."""

    def test_basic_escalation(self):
        """Failed at r=8, next in allowed_ranks above 8 and below probe=32 → r=12."""
        result = compute_escalation_candidate(
            failed_variant="energy_p90",
            failed_rank=8,
            probe_rank=32,
            allowed_ranks=[1, 2, 4, 8, 12, 16, 32],
            already_tested_ranks=set(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["actual_r"], 12)
        self.assertEqual(result["escalation_rank"], 12)
        self.assertEqual(result["original_variant"], "energy_p90")
        self.assertEqual(result["original_rank"], 8)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["policy_type"], "escalation")

    def test_naming_convention(self):
        """Escalation variant name follows {original}_esc_r{rank} pattern."""
        result = compute_escalation_candidate(
            failed_variant="knee_p90",
            failed_rank=4,
            probe_rank=32,
            allowed_ranks=[1, 2, 4, 8, 16, 32],
            already_tested_ranks=set(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["variant"], "knee_p90_esc_r8")

    def test_no_escalation_next_is_probe(self):
        """If next rank above failed equals probe_rank → None."""
        result = compute_escalation_candidate(
            failed_variant="energy_p90",
            failed_rank=8,
            probe_rank=16,
            allowed_ranks=[1, 2, 4, 8, 16],
            already_tested_ranks=set(),
        )
        self.assertIsNone(result)

    def test_no_escalation_all_higher_tested(self):
        """If all higher ranks are already tested → None."""
        result = compute_escalation_candidate(
            failed_variant="energy_p90",
            failed_rank=8,
            probe_rank=32,
            allowed_ranks=[1, 2, 4, 8, 12, 16, 32],
            already_tested_ranks={12, 16},
        )
        self.assertIsNone(result)

    def test_skip_already_tested(self):
        """Skip tested rank 12, land on 16."""
        result = compute_escalation_candidate(
            failed_variant="energy_p90",
            failed_rank=8,
            probe_rank=32,
            allowed_ranks=[1, 2, 4, 8, 12, 16, 32],
            already_tested_ranks={12},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["actual_r"], 16)

    def test_no_higher_ranks_at_all(self):
        """Failed at highest allowed rank below probe → None."""
        result = compute_escalation_candidate(
            failed_variant="energy_p90",
            failed_rank=16,
            probe_rank=16,
            allowed_ranks=[1, 2, 4, 8, 16],
            already_tested_ranks=set(),
        )
        self.assertIsNone(result)

    def test_unsorted_allowed_ranks(self):
        """allowed_ranks doesn't need to be pre-sorted."""
        result = compute_escalation_candidate(
            failed_variant="erank_p90",
            failed_rank=4,
            probe_rank=32,
            allowed_ranks=[32, 8, 2, 16, 4, 12],
            already_tested_ranks=set(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["actual_r"], 8)


# ---------------------------------------------------------------------------
# run_escalation_round
# ---------------------------------------------------------------------------


class TestRunEscalationRound(unittest.TestCase):
    """Tests for run_escalation_round()."""

    def _make_config(self, allowed_ranks=None, probe_r=32, lora_kwargs=None):
        """Helper to build a minimal bench config."""
        return {
            "compression": {
                "allowed_ranks": allowed_ranks or [1, 2, 4, 8, 12, 16, 32],
                "acc_tolerance": 0.005,
            },
            "lora": {
                "probe_r": probe_r,
                "alpha": probe_r,
                "dropout": 0.0,
                "target_modules": ["q_lin", "v_lin"],
                **(lora_kwargs or {}),
            },
        }

    def test_disabled(self):
        """Escalation disabled → empty configs."""
        configs, trace = run_escalation_round(
            variant_results={},
            verdicts={},
            compression_configs={},
            config=self._make_config(),
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={"enabled": False},
        )
        self.assertEqual(configs, {})
        self.assertFalse(trace.triggered)

    def test_pass_no_escalation(self):
        """Passing variants don't trigger escalation."""
        configs, trace = run_escalation_round(
            variant_results={"energy_p90": {"status": "completed", "rank": 8}},
            verdicts={"energy_p90": {"verdict": "PASS", "delta_vs_probe": +0.005}},
            compression_configs={"energy_p90": {"status": "ready", "actual_r": 8}},
            config=self._make_config(),
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={},
        )
        self.assertEqual(configs, {})
        self.assertFalse(trace.triggered)

    def test_near_miss_no_escalation(self):
        """Near-miss failure doesn't trigger escalation."""
        configs, trace = run_escalation_round(
            variant_results={"energy_p90": {"status": "completed", "rank": 8}},
            verdicts={"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.010}},
            compression_configs={"energy_p90": {"status": "ready", "actual_r": 8}},
            config=self._make_config(),
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={},
        )
        self.assertEqual(configs, {})
        self.assertFalse(trace.triggered)

    def test_catastrophic_triggers_escalation(self):
        """Catastrophic failure triggers escalation candidate generation."""
        configs, trace = run_escalation_round(
            variant_results={"energy_p90": {"status": "completed", "rank": 8}},
            verdicts={"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.060}},
            compression_configs={"energy_p90": {"status": "ready", "actual_r": 8}},
            config=self._make_config(),
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={},
        )
        self.assertTrue(trace.triggered)
        self.assertEqual(len(configs), 1)
        esc_name = "energy_p90_esc_r12"
        self.assertIn(esc_name, configs)
        self.assertEqual(configs[esc_name]["actual_r"], 12)
        self.assertEqual(configs[esc_name]["status"], "ready")
        # Verify LoRA config was populated
        self.assertIn("config", configs[esc_name])
        self.assertEqual(configs[esc_name]["config"]["probe_r"], 12)

    def test_escalation_trace_entry(self):
        """Verify the trace entry contains all expected fields."""
        _configs, trace = run_escalation_round(
            variant_results={"energy_p90": {"status": "completed", "rank": 8}},
            verdicts={"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.060}},
            compression_configs={"energy_p90": {"status": "ready", "actual_r": 8}},
            config=self._make_config(),
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={},
        )
        self.assertEqual(len(trace.escalation_trace), 1)
        entry = trace.escalation_trace[0]
        self.assertEqual(entry.original_variant, "energy_p90")
        self.assertEqual(entry.original_rank, 8)
        self.assertEqual(entry.failure_mode, "catastrophic")
        self.assertAlmostEqual(entry.worst_delta, -0.060)
        self.assertEqual(entry.escalated_to, "energy_p90_esc_r12")
        self.assertEqual(entry.escalation_rank, 12)
        self.assertIsNone(entry.escalation_result)  # Not yet trained

    def test_no_escalation_when_no_higher_rank_below_probe(self):
        """Catastrophic failure but no viable rank → trace entry with escalated_to=None."""
        configs, trace = run_escalation_round(
            variant_results={"energy_p90": {"status": "completed", "rank": 8}},
            verdicts={"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.060}},
            compression_configs={"energy_p90": {"status": "ready", "actual_r": 8}},
            config=self._make_config(allowed_ranks=[1, 2, 4, 8, 16], probe_r=16),
            probe_rank=16,
            acc_tolerance=0.005,
            escalation_config={},
        )
        self.assertEqual(configs, {})
        self.assertFalse(trace.triggered)
        # Trace entry still recorded (for reporting)
        self.assertEqual(len(trace.escalation_trace), 1)
        self.assertIsNone(trace.escalation_trace[0].escalated_to)


# ---------------------------------------------------------------------------
# update_escalation_trace_with_results
# ---------------------------------------------------------------------------


class TestUpdateEscalationTrace(unittest.TestCase):
    """Tests for update_escalation_trace_with_results()."""

    def _make_trace_with_entry(self, esc_name="energy_p90_esc_r12"):
        trace = EscalationTrace(
            triggered=True,
            candidates_tested=[esc_name],
            escalation_trace=[
                EscalationTraceEntry(
                    original_variant="energy_p90",
                    original_rank=8,
                    failure_mode="catastrophic",
                    worst_delta=-0.060,
                    catastrophic_threshold=-0.015,
                    escalated_to=esc_name,
                    escalation_rank=12,
                    escalation_result=None,
                )
            ],
        )
        return trace

    def test_passing_escalation(self):
        """Escalation candidate passes → final_recommendation set."""
        trace = self._make_trace_with_entry()
        trace = update_escalation_trace_with_results(
            trace=trace,
            escalation_verdicts={"energy_p90_esc_r12": {"verdict": "PASS"}},
            original_best_compression=None,
        )
        self.assertEqual(trace.escalation_trace[0].escalation_result, "PASS")
        self.assertEqual(trace.final_recommendation, "energy_p90_esc_r12")
        self.assertTrue(len(trace.recommendation_rationale) > 0)

    def test_failing_escalation_with_original_best(self):
        """Escalation fails, original best preserved."""
        trace = self._make_trace_with_entry()
        trace = update_escalation_trace_with_results(
            trace=trace,
            escalation_verdicts={"energy_p90_esc_r12": {"verdict": "FAIL"}},
            original_best_compression={"variant": "knee_p90"},
        )
        self.assertEqual(trace.escalation_trace[0].escalation_result, "FAIL")
        self.assertEqual(trace.final_recommendation, "knee_p90")

    def test_failing_escalation_no_original_best(self):
        """Escalation fails, no original best → None."""
        trace = self._make_trace_with_entry()
        trace = update_escalation_trace_with_results(
            trace=trace,
            escalation_verdicts={"energy_p90_esc_r12": {"verdict": "FAIL"}},
            original_best_compression=None,
        )
        self.assertIsNone(trace.final_recommendation)


# ---------------------------------------------------------------------------
# enrich_verdicts_with_stability
# ---------------------------------------------------------------------------


class TestEnrichVerdicts(unittest.TestCase):
    """Tests for enrich_verdicts_with_stability()."""

    def test_passing_gets_stable(self):
        """Passing variant → stability_status='stable'."""
        verdicts = {"energy_p90": {"verdict": "PASS", "delta_vs_probe": +0.005}}
        trace = EscalationTrace()
        enrich_verdicts_with_stability(verdicts, acc_tolerance=0.005, escalation_trace=trace)

        self.assertEqual(verdicts["energy_p90"]["stability_status"], "stable")
        self.assertEqual(verdicts["energy_p90"]["failure_mode"], "none")
        self.assertIsNone(verdicts["energy_p90"]["escalated_to"])

    def test_near_miss_status(self):
        """Near-miss failure → stability_status='near_miss'."""
        verdicts = {"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.010}}
        trace = EscalationTrace()
        enrich_verdicts_with_stability(verdicts, acc_tolerance=0.005, escalation_trace=trace)

        self.assertEqual(verdicts["energy_p90"]["stability_status"], "near_miss")
        self.assertEqual(verdicts["energy_p90"]["failure_mode"], "near_miss")

    def test_catastrophic_gets_unstable_with_escalation(self):
        """Catastrophic failure with escalation → unstable + escalated_to set."""
        verdicts = {"energy_p90": {"verdict": "FAIL", "delta_vs_probe": -0.060}}
        trace = EscalationTrace(
            triggered=True,
            escalation_trace=[
                EscalationTraceEntry(
                    original_variant="energy_p90",
                    original_rank=8,
                    failure_mode="catastrophic",
                    worst_delta=-0.060,
                    catastrophic_threshold=-0.015,
                    escalated_to="energy_p90_esc_r12",
                    escalation_rank=12,
                    escalation_result="PASS",
                )
            ],
        )
        enrich_verdicts_with_stability(verdicts, acc_tolerance=0.005, escalation_trace=trace)

        self.assertEqual(verdicts["energy_p90"]["stability_status"], "unstable")
        self.assertEqual(verdicts["energy_p90"]["failure_mode"], "catastrophic")
        self.assertEqual(verdicts["energy_p90"]["escalated_to"], "energy_p90_esc_r12")
        self.assertIn("Catastrophic failure", verdicts["energy_p90"]["escalation_reason"])

    def test_unknown_when_delta_is_none(self):
        """Variant with no delta → unknown stability."""
        verdicts = {"skipped_variant": {"verdict": "SKIP", "delta_vs_probe": None}}
        trace = EscalationTrace()
        enrich_verdicts_with_stability(verdicts, acc_tolerance=0.005, escalation_trace=trace)

        self.assertEqual(verdicts["skipped_variant"]["stability_status"], "unknown")


# ---------------------------------------------------------------------------
# EscalationTrace serialisation
# ---------------------------------------------------------------------------


class TestEscalationTraceSerialization(unittest.TestCase):
    """Tests for EscalationTrace.to_dict()."""

    def test_empty_trace(self):
        trace = EscalationTrace()
        d = trace.to_dict()
        self.assertFalse(d["triggered"])
        self.assertEqual(d["candidates_tested"], [])
        self.assertEqual(d["escalation_trace"], [])
        self.assertIsNone(d["final_recommendation"])

    def test_populated_trace(self):
        trace = EscalationTrace(
            triggered=True,
            candidates_tested=["energy_p90_esc_r12"],
            escalation_trace=[
                EscalationTraceEntry(
                    original_variant="energy_p90",
                    original_rank=8,
                    failure_mode="catastrophic",
                    worst_delta=-0.060,
                    catastrophic_threshold=-0.015,
                    escalated_to="energy_p90_esc_r12",
                    escalation_rank=12,
                    escalation_result="PASS",
                )
            ],
            final_recommendation="energy_p90_esc_r12",
            recommendation_rationale=["Escalated from energy_p90 (r=8) to r=12: passed."],
        )
        d = trace.to_dict()
        self.assertTrue(d["triggered"])
        self.assertEqual(len(d["escalation_trace"]), 1)
        entry = d["escalation_trace"][0]
        self.assertEqual(entry["original_variant"], "energy_p90")
        self.assertEqual(entry["escalation_result"], "PASS")
        self.assertEqual(d["final_recommendation"], "energy_p90_esc_r12")


# ---------------------------------------------------------------------------
# Integration: end-to-end escalation simulation
# ---------------------------------------------------------------------------


class TestEscalationIntegration(unittest.TestCase):
    """End-to-end test simulating a catastrophic failure → escalation → pass."""

    def test_full_escalation_flow(self):
        """Simulate: energy_p90 catastrophically fails, escalation to r=12 passes."""
        config = {
            "compression": {
                "allowed_ranks": [1, 2, 4, 8, 12, 16, 32],
                "acc_tolerance": 0.005,
            },
            "lora": {
                "probe_r": 32,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": ["q_lin", "v_lin"],
            },
        }

        # Step 1: Original candidate fails catastrophically
        variant_results = {
            "energy_p90": {
                "status": "completed",
                "rank": 8,
                "params": 100000,
                "accuracy": 0.640,
            }
        }
        verdicts = {
            "energy_p90": {
                "verdict": "FAIL",
                "delta_vs_probe": -0.060,
                "param_reduction": 0.5,
            }
        }
        compression_configs = {"energy_p90": {"status": "ready", "actual_r": 8}}

        # Step 2: Run escalation round
        esc_configs, trace = run_escalation_round(
            variant_results=variant_results,
            verdicts=verdicts,
            compression_configs=compression_configs,
            config=config,
            probe_rank=32,
            acc_tolerance=0.005,
            escalation_config={"enabled": True},
        )

        self.assertTrue(trace.triggered)
        self.assertIn("energy_p90_esc_r12", esc_configs)
        esc_cfg = esc_configs["energy_p90_esc_r12"]
        self.assertEqual(esc_cfg["actual_r"], 12)
        self.assertEqual(esc_cfg["config"]["probe_r"], 12)
        self.assertEqual(esc_cfg["config"]["alpha"], 12)

        # Step 3: Simulate escalation candidate passing
        escalation_verdicts = {
            "energy_p90_esc_r12": {
                "verdict": "PASS",
                "delta_vs_probe": -0.003,
            }
        }

        trace = update_escalation_trace_with_results(
            trace=trace,
            escalation_verdicts=escalation_verdicts,
            original_best_compression=None,
        )

        self.assertEqual(trace.final_recommendation, "energy_p90_esc_r12")
        self.assertEqual(trace.escalation_trace[0].escalation_result, "PASS")

        # Step 4: Enrich verdicts
        all_verdicts = {
            **verdicts,
            "energy_p90_esc_r12": escalation_verdicts["energy_p90_esc_r12"],
        }
        enrich_verdicts_with_stability(
            all_verdicts,
            acc_tolerance=0.005,
            escalation_trace=trace,
        )

        # Original: unstable with escalation link
        self.assertEqual(all_verdicts["energy_p90"]["stability_status"], "unstable")
        self.assertEqual(all_verdicts["energy_p90"]["escalated_to"], "energy_p90_esc_r12")

        # Escalation: stable
        self.assertEqual(all_verdicts["energy_p90_esc_r12"]["stability_status"], "stable")

        # Step 5: Verify serialisation
        trace_dict = trace.to_dict()
        self.assertTrue(trace_dict["triggered"])
        self.assertEqual(trace_dict["final_recommendation"], "energy_p90_esc_r12")
        self.assertEqual(len(trace_dict["escalation_trace"]), 1)
        self.assertEqual(trace_dict["escalation_trace"][0]["escalation_result"], "PASS")


if __name__ == "__main__":
    unittest.main()
