"""
Unit tests for decision trace and second rung logic.

Tests the pure function maybe_add_second_rung_candidates() to ensure:
1. Tier B triggers → r=16 added and rule recorded
2. Tier B does not trigger → no r=16 and non-trigger reason recorded  
3. Tier A triggers → r=24 (or computed rung) added
4. No duplication if candidate already exists
5. Schema stability: decision_trace exists and is serializable
"""

import pytest
from gradience.bench.decision_trace import (
    DecisionTrace, 
    create_decision_trace, 
    maybe_add_second_rung_candidates,
    round_to_allowed_ranks
)


def create_mock_audit_data(
    utilization_mean: float = 0.5,
    stable_rank_mean: float = 12.0,
    suggested_r_global_median: float = 20.0
) -> dict:
    """Create mock audit data for testing."""
    return {
        "audit_summary": {
            "utilization_mean": utilization_mean,
            "stable_rank_mean": stable_rank_mean,
            "current_r": 32,
        },
        "suggested_r_global_median": suggested_r_global_median,
        "suggested_r_global_90": suggested_r_global_median + 4
    }


def test_round_to_allowed_ranks():
    """Test rank rounding logic."""
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    assert round_to_allowed_ranks(15.0, allowed_ranks) == 16
    assert round_to_allowed_ranks(22.0, allowed_ranks) == 24  # Closer to 24 than 16
    assert round_to_allowed_ranks(6.0, allowed_ranks) == 4    # Equidistant, picks first in list
    assert round_to_allowed_ranks(1.0, allowed_ranks) == 2


def test_tier_b_triggers_aggressive_compression():
    """Test Tier B rule: utilization < 0.30 AND stable_rank <= probe_rank/4."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    # Create conditions where Tier B should trigger
    audit_data = create_mock_audit_data(
        utilization_mean=0.25,  # < 0.30 ✓
        stable_rank_mean=6.0,   # <= 32/4=8 ✓
        suggested_r_global_median=20.0
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = ["energy_p90", "knee_p90"]
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Should add aggressive compression candidate
    assert len(new_candidates) == 2  # Both Tier A and Tier B should trigger
    
    # Check Tier B candidate
    tier_b_candidate = next((c for c in new_candidates if c["policy_type"] == "second_rung_tier_b"), None)
    assert tier_b_candidate is not None
    assert tier_b_candidate["actual_r"] == 16  # 50% of 32
    assert tier_b_candidate["name"] == "uniform_r16"
    
    # Check rule was recorded
    tier_b_rules = [r for r in decision_trace.rules_fired if r.rule_id == "tier_b_aggressive_compression"]
    assert len(tier_b_rules) == 1
    assert "uniform_r16" in tier_b_rules[0].action


def test_tier_b_does_not_trigger():
    """Test Tier B rule does not trigger when conditions not met."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    # Create conditions where Tier B should NOT trigger
    audit_data = create_mock_audit_data(
        utilization_mean=0.40,  # >= 0.30 ✗
        stable_rank_mean=12.0,  # > 32/4=8 ✗
        suggested_r_global_median=20.0
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = ["energy_p90", "knee_p90"]
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Should not add Tier B candidate
    tier_b_candidate = next((c for c in new_candidates if c["policy_type"] == "second_rung_tier_b"), None)
    assert tier_b_candidate is None
    
    # Check rule was considered but not triggered
    tier_b_rules = [r for r in decision_trace.rules_considered if r.rule_id == "tier_b_aggressive_compression"]
    assert len(tier_b_rules) == 1
    assert "not triggered:" in tier_b_rules[0].action
    assert "utilization 0.400 >= 0.30" in tier_b_rules[0].action


def test_tier_a_triggers_moderate_compression():
    """Test Tier A rule: utilization <= 0.55 OR suggested_rank <= 0.75 * probe_rank."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    # Test utilization trigger
    audit_data = create_mock_audit_data(
        utilization_mean=0.50,  # <= 0.55 ✓
        stable_rank_mean=20.0,  # Not relevant for Tier A
        suggested_r_global_median=30.0  # > 0.75*32=24 (won't trigger by itself)
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = ["energy_p90", "knee_p90"]
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Should add moderate compression candidate
    tier_a_candidate = next((c for c in new_candidates if c["policy_type"] == "second_rung_tier_a"), None)
    assert tier_a_candidate is not None
    assert tier_a_candidate["actual_r"] == 24  # 75% of 32 rounded to allowed
    assert tier_a_candidate["name"] == "uniform_r24"
    
    # Check rule was recorded
    tier_a_rules = [r for r in decision_trace.rules_fired if r.rule_id == "tier_a_moderate_compression"]
    assert len(tier_a_rules) == 1
    assert "uniform_r24" in tier_a_rules[0].action


def test_tier_a_triggers_by_suggested_rank():
    """Test Tier A triggers via suggested_rank <= 0.75 * probe_rank."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    # Test suggested rank trigger
    audit_data = create_mock_audit_data(
        utilization_mean=0.70,  # > 0.55 (won't trigger by itself) 
        stable_rank_mean=20.0,  
        suggested_r_global_median=20.0  # <= 0.75*32=24 ✓
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = ["energy_p90", "knee_p90"]
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Should add moderate compression candidate
    tier_a_candidate = next((c for c in new_candidates if c["policy_type"] == "second_rung_tier_a"), None)
    assert tier_a_candidate is not None
    assert tier_a_candidate["actual_r"] == 24


def test_no_duplication_if_candidate_exists():
    """Test that candidates are not added if they already exist."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    # Create conditions where both tiers would trigger
    audit_data = create_mock_audit_data(
        utilization_mean=0.25,  # Triggers both tiers
        stable_rank_mean=6.0,   # Triggers Tier B
        suggested_r_global_median=20.0
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    
    # Pre-existing candidates that would conflict
    existing_candidates = ["energy_p90", "knee_p90", "uniform_r24", "uniform_r16"]
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Should not add any new candidates due to conflicts
    assert len(new_candidates) == 0
    
    # Check rules were considered but not triggered due to duplicates
    tier_a_rules = [r for r in decision_trace.rules_considered if r.rule_id == "tier_a_moderate_compression"]
    tier_b_rules = [r for r in decision_trace.rules_considered if r.rule_id == "tier_b_aggressive_compression"]
    
    assert len(tier_a_rules) == 1
    assert "uniform_r24 already exists" in tier_a_rules[0].action
    
    assert len(tier_b_rules) == 1
    assert "uniform_r16 already exists" in tier_b_rules[0].action


def test_decision_trace_schema_stability():
    """Test that DecisionTrace is JSON-serializable and has stable schema."""
    probe_rank = 32
    allowed_ranks = [2, 4, 8, 16, 24, 32]
    
    audit_data = create_mock_audit_data(
        utilization_mean=0.25,
        stable_rank_mean=6.0,
        suggested_r_global_median=20.0
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = ["energy_p90"]
    
    # Run the logic to populate decision trace
    maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Test serialization
    trace_dict = decision_trace.to_dict()
    
    # Check required schema fields
    assert "probe_rank" in trace_dict
    assert "audit_metrics" in trace_dict
    assert "rules_fired" in trace_dict
    assert "rules_considered" in trace_dict
    
    assert trace_dict["probe_rank"] == 32
    assert isinstance(trace_dict["audit_metrics"], dict)
    assert isinstance(trace_dict["rules_fired"], list)
    assert isinstance(trace_dict["rules_considered"], list)
    
    # Test deserialization round-trip
    restored_trace = DecisionTrace.from_dict(trace_dict)
    assert restored_trace.probe_rank == decision_trace.probe_rank
    assert restored_trace.audit_metrics == decision_trace.audit_metrics
    assert len(restored_trace.rules_fired) == len(decision_trace.rules_fired)
    
    # Ensure JSON serialization works
    import json
    json_str = json.dumps(trace_dict)  # Should not raise
    parsed_back = json.loads(json_str)
    assert parsed_back["probe_rank"] == 32


def test_create_decision_trace_from_audit_data():
    """Test decision trace creation from various audit data formats."""
    probe_rank = 32
    
    # Test with audit_summary format
    audit_data = {
        "audit_summary": {
            "utilization_mean": 0.45,
            "stable_rank_mean": 15.2,
            "current_r": 32,
        },
        "suggested_r_global_median": 18,
        "suggested_r_global_90": 22
    }
    
    trace = create_decision_trace(probe_rank, audit_data)
    
    assert trace.probe_rank == 32
    assert trace.audit_metrics["utilization_mean"] == 0.45
    assert trace.audit_metrics["stable_rank_mean"] == 15.2
    assert trace.audit_metrics["current_r"] == 32
    assert trace.audit_metrics["suggested_r_global_median"] == 18
    assert trace.audit_metrics["suggested_r_global_90"] == 22


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    probe_rank = 8  # Small probe rank
    allowed_ranks = [2, 4, 8, 16]
    
    # Edge case: very small utilization and stable rank
    audit_data = create_mock_audit_data(
        utilization_mean=0.05,  # Very low
        stable_rank_mean=1.0,   # Very low
        suggested_r_global_median=2.0
    )
    
    decision_trace = create_decision_trace(probe_rank, audit_data)
    existing_candidates = []
    
    new_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates,
        decision_trace=decision_trace
    )
    
    # Both tiers should trigger, but check computed ranks make sense
    assert len(new_candidates) <= 2
    
    # All actual ranks should be valid and less than probe rank
    for candidate in new_candidates:
        assert candidate["actual_r"] in allowed_ranks
        assert candidate["actual_r"] < probe_rank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])