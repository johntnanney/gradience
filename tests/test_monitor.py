"""Tests for gradience.vnext.monitor — training monitor with heuristic rules."""

from __future__ import annotations

import json

import pytest

from gradience.vnext.monitor import MonitorAlert, MonitorConfig, TrainingMonitor
from gradience.vnext.monitor_replay import replay_telemetry

# ---------------------------------------------------------------------------
# MonitorConfig
# ---------------------------------------------------------------------------


class TestMonitorConfig:
    def test_defaults(self):
        c = MonitorConfig()
        assert c.window_size == 50
        assert c.loss_plateau_patience == 5
        assert c.enable_loss_plateau is True

    def test_custom(self):
        c = MonitorConfig(window_size=100, loss_plateau_patience=10)
        assert c.window_size == 100
        assert c.loss_plateau_patience == 10


# ---------------------------------------------------------------------------
# MonitorAlert
# ---------------------------------------------------------------------------


class TestMonitorAlert:
    def test_to_dict_serializable(self):
        a = MonitorAlert(
            step=100,
            code="TEST",
            severity="info",
            message="test",
            details={"x": 1},
            recommendation="do something",
        )
        d = a.to_dict()
        json_str = json.dumps(d)
        assert "TEST" in json_str

    def test_frozen(self):
        a = MonitorAlert(
            step=100,
            code="TEST",
            severity="info",
            message="test",
            details={},
            recommendation="",
        )
        with pytest.raises(AttributeError):
            a.step = 200


# ---------------------------------------------------------------------------
# Loss plateau detection
# ---------------------------------------------------------------------------


class TestLossPlateau:
    def test_no_alert_when_loss_decreasing(self):
        """Steadily decreasing loss should not trigger plateau."""
        monitor = TrainingMonitor(MonitorConfig(loss_plateau_patience=3))
        for i in range(20):
            alerts = monitor.update(i, {"loss": 1.0 - i * 0.05})
        plateau_alerts = [a for a in alerts if a.code == "LOSS_PLATEAU"]
        assert len(plateau_alerts) == 0

    def test_alert_on_flat_loss(self):
        """Constant loss should trigger plateau after patience steps."""
        config = MonitorConfig(
            loss_plateau_patience=3,
            loss_plateau_delta=0.01,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(10):
            alerts = monitor.update(i, {"loss": 0.5})
            all_alerts.extend(alerts)

        plateau_alerts = [a for a in all_alerts if a.code == "LOSS_PLATEAU"]
        assert len(plateau_alerts) > 0
        # First plateau should fire around step 3 (patience=3)
        assert plateau_alerts[0].step <= 5

    def test_no_alert_when_disabled(self):
        """Disabled rule should not fire."""
        config = MonitorConfig(enable_loss_plateau=False)
        monitor = TrainingMonitor(config)
        for i in range(20):
            monitor.update(i, {"loss": 0.5})
        # Would have been a plateau, but rule is disabled
        assert monitor._loss_stale_count == 0  # counter not updated

    def test_counter_resets_on_change(self):
        """Big loss change should reset the stale counter."""
        config = MonitorConfig(loss_plateau_patience=5, loss_plateau_delta=0.01)
        monitor = TrainingMonitor(config)
        # 3 flat steps
        for i in range(3):
            monitor.update(i, {"loss": 0.5})
        # Big change
        monitor.update(3, {"loss": 0.3})
        assert monitor._loss_stale_count == 0


# ---------------------------------------------------------------------------
# Gradient plateau detection
# ---------------------------------------------------------------------------


class TestGradPlateau:
    def test_flat_grad_triggers_alert(self):
        config = MonitorConfig(
            grad_plateau_patience=3,
            grad_plateau_delta=0.02,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(10):
            alerts = monitor.update(i, {"grad_norm": 1.0})
            all_alerts.extend(alerts)

        grad_alerts = [a for a in all_alerts if a.code == "GRAD_PLATEAU"]
        assert len(grad_alerts) > 0


# ---------------------------------------------------------------------------
# Gradient spike detection
# ---------------------------------------------------------------------------


class TestGradSpike:
    def test_spike_detected(self):
        """A sudden gradient spike should trigger an alert."""
        config = MonitorConfig(
            grad_spike_multiplier=3.0,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        # Build up a stable history
        for i in range(10):
            monitor.update(i, {"grad_norm": 1.0})
        # Spike!
        alerts = monitor.update(10, {"grad_norm": 10.0})

        spike_alerts = [a for a in alerts if a.code == "GRAD_SPIKE"]
        assert len(spike_alerts) == 1
        assert spike_alerts[0].details["spike_ratio"] > 3.0

    def test_no_spike_for_gradual_increase(self):
        """Gradual increase should not trigger spike."""
        config = MonitorConfig(grad_spike_multiplier=5.0)
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(20):
            alerts = monitor.update(i, {"grad_norm": 1.0 + i * 0.1})
            all_alerts.extend(alerts)

        spike_alerts = [a for a in all_alerts if a.code == "GRAD_SPIKE"]
        assert len(spike_alerts) == 0

    def test_no_spike_with_short_history(self):
        """Too few steps should not trigger spike detection."""
        config = MonitorConfig(grad_spike_multiplier=2.0)
        monitor = TrainingMonitor(config)
        alerts = monitor.update(0, {"grad_norm": 100.0})
        spike_alerts = [a for a in alerts if a.code == "GRAD_SPIKE"]
        assert len(spike_alerts) == 0


# ---------------------------------------------------------------------------
# Eval plateau detection
# ---------------------------------------------------------------------------


class TestEvalPlateau:
    def test_eval_plateau_when_no_improvement(self):
        """Eval loss that never improves should trigger plateau."""
        config = MonitorConfig(eval_patience=3, alert_cooldown_steps=0)
        monitor = TrainingMonitor(config)
        all_alerts = []
        # First eval (baseline)
        all_alerts.extend(monitor.update_eval(0, {"loss": 0.5}))
        # 4 more evals with no improvement
        for i in range(1, 5):
            all_alerts.extend(monitor.update_eval(i * 100, {"loss": 0.6}))

        eval_alerts = [a for a in all_alerts if a.code == "EVAL_PLATEAU"]
        assert len(eval_alerts) > 0

    def test_no_eval_plateau_when_improving(self):
        """Improving eval loss should not trigger plateau."""
        config = MonitorConfig(eval_patience=3, alert_cooldown_steps=0)
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(10):
            all_alerts.extend(monitor.update_eval(i * 100, {"loss": 1.0 - i * 0.05}))

        eval_alerts = [a for a in all_alerts if a.code == "EVAL_PLATEAU"]
        assert len(eval_alerts) == 0


# ---------------------------------------------------------------------------
# Utilization check
# ---------------------------------------------------------------------------


class TestUtilizationCheck:
    def test_low_utilization_alert(self):
        config = MonitorConfig(
            utilization_warn_threshold=0.25,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        alerts = monitor.update(100, {"loss": 0.5, "utilization": 0.10})

        util_alerts = [a for a in alerts if a.code == "UTILIZATION_LOW"]
        assert len(util_alerts) == 1
        assert util_alerts[0].details["utilization"] == 0.10

    def test_ok_utilization_no_alert(self):
        config = MonitorConfig(utilization_warn_threshold=0.25)
        monitor = TrainingMonitor(config)
        alerts = monitor.update(100, {"loss": 0.5, "utilization": 0.50})
        util_alerts = [a for a in alerts if a.code == "UTILIZATION_LOW"]
        assert len(util_alerts) == 0


# ---------------------------------------------------------------------------
# CONSIDER_STOPPING meta-rule
# ---------------------------------------------------------------------------


class TestConsiderStopping:
    def test_fires_when_loss_and_grad_plateau(self):
        """Both loss and grad plateauing should trigger CONSIDER_STOPPING."""
        config = MonitorConfig(
            loss_plateau_patience=2,
            grad_plateau_patience=2,
            loss_plateau_delta=0.01,
            grad_plateau_delta=0.02,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(10):
            alerts = monitor.update(i, {"loss": 0.5, "grad_norm": 1.0})
            all_alerts.extend(alerts)

        stop_alerts = [a for a in all_alerts if a.code == "CONSIDER_STOPPING"]
        assert len(stop_alerts) > 0

    def test_fires_when_eval_and_loss_plateau(self):
        """Eval plateau + loss plateau should trigger CONSIDER_STOPPING."""
        config = MonitorConfig(
            loss_plateau_patience=2,
            loss_plateau_delta=0.01,
            eval_patience=2,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []

        # Build up eval plateau
        monitor.update_eval(0, {"loss": 0.5})
        for i in range(1, 4):
            all_alerts.extend(monitor.update_eval(i * 100, {"loss": 0.6}))

        # Now feed flat train loss (same step as last eval)
        for i in range(5):
            all_alerts.extend(monitor.update(400 + i, {"loss": 0.3}))

        # Then trigger eval check again
        all_alerts.extend(monitor.update_eval(500, {"loss": 0.6}))

        stop_alerts = [a for a in all_alerts if a.code == "CONSIDER_STOPPING"]
        assert len(stop_alerts) > 0

    def test_does_not_fire_with_single_plateau(self):
        """A single plateau signal should not trigger CONSIDER_STOPPING."""
        config = MonitorConfig(
            loss_plateau_patience=2,
            loss_plateau_delta=0.01,
            alert_cooldown_steps=0,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []
        # Flat loss but changing grad_norm
        for i in range(10):
            alerts = monitor.update(i, {"loss": 0.5, "grad_norm": 1.0 + i * 0.5})
            all_alerts.extend(alerts)

        stop_alerts = [a for a in all_alerts if a.code == "CONSIDER_STOPPING"]
        assert len(stop_alerts) == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_suppresses_repeated_alerts(self):
        """Same alert should be suppressed within cooldown window."""
        config = MonitorConfig(
            loss_plateau_patience=2,
            loss_plateau_delta=0.01,
            alert_cooldown_steps=10,
        )
        monitor = TrainingMonitor(config)
        all_alerts = []
        for i in range(20):
            alerts = monitor.update(i, {"loss": 0.5})
            all_alerts.extend(alerts)

        plateau_alerts = [a for a in all_alerts if a.code == "LOSS_PLATEAU"]
        # With cooldown=10 and 20 steps, should fire at most 2 times
        assert len(plateau_alerts) <= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_loss_ignored(self):
        """NaN loss should be ignored, not crash."""
        monitor = TrainingMonitor()
        alerts = monitor.update(0, {"loss": float("nan")})
        assert isinstance(alerts, list)
        assert monitor.n_steps == 1  # step counted

    def test_missing_metrics(self):
        """Missing metrics should not crash."""
        monitor = TrainingMonitor()
        alerts = monitor.update(0, {})
        assert isinstance(alerts, list)

    def test_empty_monitor(self):
        """Fresh monitor with no data should not crash on any rule."""
        monitor = TrainingMonitor()
        assert monitor.n_steps == 0
        assert monitor.n_evals == 0

    def test_single_step(self):
        """Single step should not trigger any alerts."""
        monitor = TrainingMonitor()
        alerts = monitor.update(0, {"loss": 1.0, "grad_norm": 1.0})
        # No plateau possible with 1 step
        plateau_alerts = [a for a in alerts if "PLATEAU" in a.code]
        assert len(plateau_alerts) == 0


# ---------------------------------------------------------------------------
# Telemetry replay
# ---------------------------------------------------------------------------


class TestReplayTelemetry:
    def test_replay_v1_schema(self, tmp_path):
        """Replay Gradience v1 schema JSONL."""
        telemetry = tmp_path / "test_run.jsonl"
        records = []
        for i in range(20):
            records.append(
                json.dumps(
                    {
                        "schema": "gradience.vnext.telemetry/v1",
                        "event": "train_step",
                        "step": i * 10,
                        "loss": 0.5,
                        "grad_norm": 1.0,
                    }
                )
            )
        telemetry.write_text("\n".join(records))

        result = replay_telemetry(
            telemetry,
            MonitorConfig(loss_plateau_patience=3, alert_cooldown_steps=0),
        )
        assert result.n_steps_processed == 20
        assert isinstance(result.alerts, list)

    def test_replay_flat_format(self, tmp_path):
        """Replay Study 7 flat format JSONL."""
        telemetry = tmp_path / "study7.jsonl"
        records = []
        for i in range(20):
            records.append(
                json.dumps(
                    {
                        "step": i * 200,
                        "train_loss": 1.0 - i * 0.04,
                        "val_loss": 5.0,
                        "grad_norm": 1.0,
                    }
                )
            )
        telemetry.write_text("\n".join(records))

        result = replay_telemetry(telemetry)
        assert result.n_steps_processed == 20
        assert result.n_evals_processed == 20  # val_loss counts as eval
        assert result.first_step == 0
        assert result.last_step == 3800

    def test_replay_empty_file(self, tmp_path):
        """Empty file should not crash."""
        telemetry = tmp_path / "empty.jsonl"
        telemetry.write_text("")
        result = replay_telemetry(telemetry)
        assert result.n_steps_processed == 0
        assert len(result.alerts) == 0

    def test_replay_result_summary(self, tmp_path):
        """ReplayResult.summary() should return a string."""
        telemetry = tmp_path / "test.jsonl"
        records = [json.dumps({"step": i, "train_loss": 0.5}) for i in range(5)]
        telemetry.write_text("\n".join(records))
        result = replay_telemetry(telemetry)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Replay" in summary

    def test_would_have_stopped(self, tmp_path):
        """Verify would_have_stopped detection."""
        telemetry = tmp_path / "converged.jsonl"
        records = []
        # Flat loss + flat grad + flat eval → should trigger CONSIDER_STOPPING
        for i in range(30):
            records.append(
                json.dumps(
                    {
                        "step": i,
                        "train_loss": 0.5,
                        "grad_norm": 1.0,
                        "val_loss": 2.0,
                    }
                )
            )
        telemetry.write_text("\n".join(records))

        config = MonitorConfig(
            loss_plateau_patience=3,
            grad_plateau_patience=3,
            eval_patience=3,
            alert_cooldown_steps=0,
        )
        result = replay_telemetry(telemetry, config)
        assert result.would_have_stopped is True
        assert result.would_have_stopped_step is not None
