"""
Gradience Unit Tests

Tests core functionality without requiring PyTorch/transformers.
"""

import math
import pytest
import tempfile
import os
import json

from gradience.spectral import (
    SpectralAnalyzer,
    SpectralSnapshot,
    RiskLevel,
)
from gradience.guard import (
    Guard,
    GuardConfig,
    IntegrityStatus,
    IntegrityTracker,
)
from gradience.telemetry import TelemetryWriter, SummaryGenerator


class TestIntegrityTracker:
    def test_initial_state(self):
        t = IntegrityTracker()
        assert t.status == IntegrityStatus.OK
        assert t.weights_finite is True
        assert t.loss_valid is True
    
    def test_detect_nonfinite_weights(self):
        t = IntegrityTracker()
        t.check_weights(has_nonfinite=True)
        assert t.status == IntegrityStatus.CORRUPT
        assert t.weights_finite is False
        assert t.corruption_count == 1
    
    def test_detect_invalid_loss(self):
        t = IntegrityTracker()
        t.check_loss(float('nan'))
        assert t.status == IntegrityStatus.CORRUPT
        assert t.loss_valid is False
    
    def test_recovery(self):
        t = IntegrityTracker()
        t.check_weights(has_nonfinite=True)
        t.mark_recovered("test rollback")
        assert t.status == IntegrityStatus.RECOVERED
        assert t.recovery_count == 1


class TestSpectralAnalyzer:
    def test_slope_increasing(self):
        a = SpectralAnalyzer(window_size=10)
        
        for i in range(10):
            s = SpectralSnapshot(
                step=i, timestamp=float(i),
                sigma_max=1.0, sigma_min=0.1,
                kappa_tilde=10.0 + i * 0.5,
            )
            a.add_observation(s)
        
        slope = a.compute_slope()
        assert slope is not None
        assert abs(slope - 0.5) < 0.01
    
    def test_slope_stable(self):
        a = SpectralAnalyzer(window_size=10)
        
        for i in range(10):
            s = SpectralSnapshot(
                step=i, timestamp=float(i),
                sigma_max=1.0, sigma_min=0.1,
                kappa_tilde=10.0,
            )
            a.add_observation(s)
        
        slope = a.compute_slope()
        assert abs(slope) < 0.001
    
    def test_risk_stable(self):
        a = SpectralAnalyzer(window_size=10)
        
        for i in range(10):
            s = SpectralSnapshot(
                step=i, timestamp=float(i),
                sigma_max=1.0, sigma_min=0.1,
                kappa_tilde=10.0 + i * 0.0001,
            )
            a.add_observation(s)
        
        risk, _ = a.assess_risk()
        assert risk == RiskLevel.STABLE
    
    def test_risk_danger(self):
        a = SpectralAnalyzer(window_size=10, volatility_threshold=1.0)
        
        for i in range(10):
            s = SpectralSnapshot(
                step=i, timestamp=float(i),
                sigma_max=1.0, sigma_min=0.1,
                kappa_tilde=10.0 + i * 0.01,  # slope = 0.01 > 0.004
            )
            a.add_observation(s)
        
        risk, _ = a.assess_risk()
        assert risk == RiskLevel.DANGER


class TestTelemetry:
    def test_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            w = TelemetryWriter(out_dir=tmpdir)
            w.open()
            
            w.log_event("test", {"value": 42})
            w.log_telemetry(step=10, loss=2.5)
            
            w.close()
            
            # Read back
            filepath = os.path.join(tmpdir, "telemetry.jsonl")
            with open(filepath) as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            r1 = json.loads(lines[0])
            assert r1["event"] == "test"
            assert r1["value"] == 42
    
    def test_summary(self):
        s = SummaryGenerator()
        s.total_steps = 100
        s.final_loss = 1.5
        
        result = s.generate()
        assert result["total_steps"] == 100
        assert result["final_loss"] == 1.5


class TestGuard:
    def test_thrash_detection(self):
        config = GuardConfig(
            max_rollbacks_per_window=3,
            rollback_window=100,
        )
        g = Guard(config=config)
        
        # Simulate rapid rollbacks
        g._rollback_history.append(10)
        g._rollback_history.append(20)
        g._rollback_history.append(30)
        
        assert g._is_thrashing() is True
    
    def test_no_thrash(self):
        config = GuardConfig(
            max_rollbacks_per_window=3,
            rollback_window=100,
        )
        g = Guard(config=config)
        
        # Spread out rollbacks
        g._rollback_history.append(10)
        g._rollback_history.append(200)
        g._rollback_history.append(400)
        
        assert g._is_thrashing() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
