"""
CPU-only Guard validation tests using the reusable driver.

These tests validate Guard functionality with specific scenarios
using fake logs for deterministic, fast execution.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gradience.vnext.integrations.hf import GradienceCallback
from tests.helpers.hf_callback_driver import HFCallbackDriver, LogEvent


class TestGuardCPUValidation(unittest.TestCase):
    """CPU-only Guard validation tests."""
    
    def test_1_grad_explosion_triggers_rollback(self):
        """
        Test 1: Grad explosion triggers rollback (CPU-only)
        
        Setup:
        - enable_guard=True
        - grad_threshold=100.0
        - snapshot_every=1
        - cooldown_steps=0
        - max_rollbacks=10
        
        Sequence:
        - step 1: logs={"loss": 1.0, "grad_norm": 1.0} → ok
        - step 2: logs={"loss": 1.0, "grad_norm": 1e6} → trigger grad_explosion
        
        Assertions:
        - JSONL contains alert code GUARD_TRIGGERED with trigger grad_explosion
        - JSONL contains alert code GUARD_ROLLBACK
        - JSONL contains metrics(kind="guard") with action rollback and restored_step
        """
        print("\n" + "="*60)
        print("Test 1: Grad explosion triggers rollback (CPU-only)")
        print("="*60)
        
        # Setup Guard configuration
        guard_config = {
            "enable_guard": True,
            "guard_grad_threshold": 100.0,
            "guard_snapshot_every": 1,
            "guard_cooldown_steps": 0,
            "guard_max_rollbacks": 10,
        }
        
        print(f"Guard configuration:")
        for key, value in guard_config.items():
            print(f"  {key}: {value}")
        
        # Create driver
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=guard_config
        )
        
        try:
            # Define exact sequence from test specification
            log_events = [
                LogEvent(step=1, loss=1.0, grad_norm=1.0),     # Normal step → ok
                LogEvent(step=2, loss=1.0, grad_norm=1e6),     # Explosion → trigger grad_explosion
            ]
            
            print(f"\nLog sequence:")
            for event in log_events:
                print(f"  step {event.step}: loss={event.loss}, grad_norm={event.grad_norm}")
            
            # Run the scenario
            result = driver.run_events(log_events, scenario_name="test_1_grad_explosion")
            
            print(f"\nResults:")
            print(f"  Total events: {len(result.events)}")
            print(f"  Guard alerts: {len(result.guard_alerts)}")
            print(f"  Guard metrics: {len(result.guard_metrics)}")
            
            # === ASSERTION 1: GUARD_TRIGGERED with trigger grad_explosion ===
            print(f"\n1. Checking GUARD_TRIGGERED alert...")
            triggered_alert = result.assert_alert_present("GUARD_TRIGGERED")
            
            # Verify trigger type is grad_explosion
            metadata = triggered_alert.get("metadata", {})
            trigger_type = metadata.get("trigger")
            self.assertEqual(trigger_type, "grad_explosion", 
                           f"Expected trigger 'grad_explosion', got '{trigger_type}'")
            print(f"   ✓ GUARD_TRIGGERED found with trigger=grad_explosion")
            print(f"   ✓ Message: {triggered_alert.get('message')}")
            
            # === ASSERTION 2: GUARD_ROLLBACK ===
            print(f"\n2. Checking GUARD_ROLLBACK alert...")
            rollback_alert = result.assert_alert_present("GUARD_ROLLBACK")
            print(f"   ✓ GUARD_ROLLBACK found")
            print(f"   ✓ Message: {rollback_alert.get('message')}")
            
            # === ASSERTION 3: metrics(kind="guard") with action rollback and restored_step ===
            print(f"\n3. Checking Guard metrics...")
            
            # Find rollback metrics
            rollback_metrics = result.get_metrics_by_action("rollback")
            self.assertGreater(len(rollback_metrics), 0, "Should have rollback metrics")
            
            rollback_metric = rollback_metrics[0]
            metrics_data = rollback_metric.get("metrics", {})
            
            # Verify action is rollback
            action = metrics_data.get("action")
            self.assertEqual(action, "rollback", f"Expected action 'rollback', got '{action}'")
            print(f"   ✓ Found metrics with action=rollback")
            
            # Verify restored_step is present and valid
            restored_step = metrics_data.get("restored_step")
            self.assertIsNotNone(restored_step, "restored_step should be present")
            self.assertIsInstance(restored_step, int, "restored_step should be an integer")
            self.assertEqual(restored_step, 1, f"Expected restored_step=1, got {restored_step}")
            print(f"   ✓ restored_step={restored_step}")
            
            # Verify kind is guard
            kind = rollback_metric.get("kind")
            self.assertEqual(kind, "guard", f"Expected kind 'guard', got '{kind}'")
            print(f"   ✓ Metrics kind=guard")
            
            # Additional verification: rollback count
            result.assert_rollback_count(1)
            print(f"   ✓ Rollback count verified: 1")
            
            # Show complete metrics for debugging
            print(f"\nRollback metrics details:")
            for key, value in metrics_data.items():
                print(f"  {key}: {value}")
            
            # === SUMMARY ===
            print(f"\n" + "="*60)
            print("✅ TEST 1 PASSED!")
            print("="*60)
            print(f"Summary:")
            print(f"  • Guard detected grad_norm=1e6 > threshold=100.0")
            print(f"  • Triggered grad_explosion at step 2")
            print(f"  • Rolled back to step {restored_step}")
            print(f"  • All required telemetry events present")
            print(f"  • JSONL file: {result.telemetry_file}")
            
        finally:
            driver.cleanup()

    def test_2_cooldown_lockout_aborts(self):
        """
        Test 2: Cooldown lockout aborts (CPU-only)
        
        Setup:
        - enable_guard=True
        - cooldown_steps=20
        - max_rollbacks=10
        - snapshot_every=1
        
        Sequence:
        - step 10: logs={"loss": nan} → rollback ok
        - step 15: logs={"loss": nan} → should abort (cooldown)
        
        Assertions:
        - JSONL contains GUARD_ROLLBACK once
        - JSONL contains GUARD_ABORT at step 15
        - control.should_training_stop set True (if your callback sets it)
        - metrics show action: abort with reason cooldown (if you log it)
        """
        print("\n" + "="*60)
        print("Test 2: Cooldown lockout aborts (CPU-only)")
        print("="*60)
        
        # Setup Guard configuration
        guard_config = {
            "enable_guard": True,
            "guard_cooldown_steps": 20,
            "guard_max_rollbacks": 10,
            "guard_snapshot_every": 1,
        }
        
        print(f"Guard configuration:")
        for key, value in guard_config.items():
            print(f"  {key}: {value}")
        
        # Create driver
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=guard_config
        )
        
        try:
            # Define exact sequence from test specification
            log_events = [
                # Steps 1-9: normal progression to establish snapshots
                *[LogEvent(step=i, loss=2.0, grad_norm=1.0) for i in range(1, 10)],
                LogEvent(step=10, loss=float('nan'), grad_norm=1.0),  # NaN loss → rollback ok
                # Steps 11-14: normal after rollback
                *[LogEvent(step=i, loss=2.0, grad_norm=1.0) for i in range(11, 15)],
                LogEvent(step=15, loss=float('nan'), grad_norm=1.0),  # NaN loss → should abort (cooldown)
            ]
            
            print(f"\nLog sequence:")
            for event in log_events:
                if event.step in [10, 15]:
                    print(f"  step {event.step}: loss={event.loss}, grad_norm={event.grad_norm} ← trigger")
                else:
                    print(f"  step {event.step}: loss={event.loss}, grad_norm={event.grad_norm}")
            
            # Run the scenario
            result = driver.run_events(log_events, scenario_name="test_2_cooldown_lockout")
            
            print(f"\nResults:")
            print(f"  Total events: {len(result.events)}")
            print(f"  Guard alerts: {len(result.guard_alerts)}")
            print(f"  Guard metrics: {len(result.guard_metrics)}")
            
            # === ASSERTION 1: GUARD_ROLLBACK once ===
            print(f"\n1. Checking GUARD_ROLLBACK count...")
            rollback_alerts = [e for e in result.guard_alerts if e.get("code") == "GUARD_ROLLBACK"]
            self.assertEqual(len(rollback_alerts), 1, 
                           f"Expected 1 GUARD_ROLLBACK, got {len(rollback_alerts)}")
            
            rollback_alert = rollback_alerts[0]
            rollback_step = rollback_alert.get("step") or rollback_alert.get("metadata", {}).get("step")
            print(f"   ✓ Found exactly 1 GUARD_ROLLBACK at step {rollback_step}")
            print(f"   ✓ Message: {rollback_alert.get('message')}")
            
            # === ASSERTION 2: GUARD_ABORT at step 15 ===
            print(f"\n2. Checking GUARD_ABORT at step 15...")
            abort_alert = result.assert_alert_present("GUARD_ABORT")
            abort_step = abort_alert.get("step") or abort_alert.get("metadata", {}).get("step")
            self.assertEqual(abort_step, 15, f"Expected GUARD_ABORT at step 15, got step {abort_step}")
            print(f"   ✓ GUARD_ABORT found at step {abort_step}")
            print(f"   ✓ Message: {abort_alert.get('message')}")
            
            # === ASSERTION 3: control.should_training_stop set True ===
            print(f"\n3. Checking control.should_training_stop...")
            # Check if callback set the training stop flag
            should_stop = driver.mock_control.should_training_stop
            if should_stop:
                print(f"   ✓ control.should_training_stop = {should_stop}")
            else:
                print(f"   ⚠ control.should_training_stop = {should_stop} (callback may not set this)")
            
            # === ASSERTION 4: metrics show action: abort with reason cooldown ===
            print(f"\n4. Checking abort metrics...")
            
            # Find abort metrics
            abort_metrics = result.get_metrics_by_action("abort")
            self.assertGreater(len(abort_metrics), 0, "Should have abort metrics")
            
            abort_metric = abort_metrics[0]
            metrics_data = abort_metric.get("metrics", {})
            
            # Verify action is abort
            action = metrics_data.get("action")
            self.assertEqual(action, "abort", f"Expected action 'abort', got '{action}'")
            print(f"   ✓ Found metrics with action=abort")
            
            # Verify reason is cooldown
            reason = metrics_data.get("reason")
            if reason:
                self.assertIn("cooldown", reason.lower(), f"Expected reason to contain 'cooldown', got '{reason}'")
                print(f"   ✓ reason='{reason}'")
            else:
                # Look for cooldown context in other fields
                abort_context = abort_metric.get("context", {})
                print(f"   ⚠ No 'reason' field, checking context: {abort_context}")
            
            # Verify kind is guard
            kind = abort_metric.get("kind")
            self.assertEqual(kind, "guard", f"Expected kind 'guard', got '{kind}'")
            print(f"   ✓ Metrics kind=guard")
            
            # Show complete abort metrics for debugging
            print(f"\nAbort metrics details:")
            for key, value in metrics_data.items():
                print(f"  {key}: {value}")
            
            # === SUMMARY ===
            print(f"\n" + "="*60)
            print("✅ TEST 2 PASSED!")
            print("="*60)
            print(f"Summary:")
            print(f"  • First NaN at step 10 triggered rollback")
            print(f"  • Second NaN at step 15 aborted due to cooldown protection")
            print(f"  • Cooldown window prevents rollbacks within 20 steps")
            print(f"  • Anti-thrash protection working correctly")
            print(f"  • JSONL file: {result.telemetry_file}")
            
        finally:
            driver.cleanup()

    def test_3_window_max_rollbacks_aborts(self):
        """
        Test 3: Window max rollbacks aborts (CPU-only)
        
        Setup:
        - cooldown_steps=0
        - max_rollbacks=1
        - window_steps=200
        - snapshot_every=1
        
        Sequence:
        - step 10: trigger → rollback ok
        - step 11: trigger → abort (max rollbacks in window)
        
        Assertions:
        - GUARD_ROLLBACK once
        - GUARD_ABORT second time, with reason max rollbacks
        """
        print("\n" + "="*60)
        print("Test 3: Window max rollbacks aborts (CPU-only)")
        print("="*60)
        
        # Setup Guard configuration
        guard_config = {
            "enable_guard": True,
            "guard_cooldown_steps": 0,
            "guard_max_rollbacks": 1,
            "guard_window_steps": 200,
            "guard_snapshot_every": 1,
            "guard_grad_threshold": 100.0,  # Add threshold for grad_norm triggers
        }
        
        print(f"Guard configuration:")
        for key, value in guard_config.items():
            print(f"  {key}: {value}")
        
        # Create driver
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=guard_config
        )
        
        try:
            # Define exact sequence from test specification
            log_events = [
                # Steps 1-9: normal progression to establish snapshots
                *[LogEvent(step=i, loss=2.0, grad_norm=1.0) for i in range(1, 10)],
                LogEvent(step=10, loss=2.0, grad_norm=500.0),  # High grad_norm → rollback ok
                LogEvent(step=11, loss=2.0, grad_norm=800.0),  # High grad_norm → abort (max rollbacks)
            ]
            
            print(f"\nLog sequence:")
            for event in log_events:
                if event.step in [10, 11]:
                    print(f"  step {event.step}: loss={event.loss}, grad_norm={event.grad_norm} ← trigger")
                else:
                    print(f"  step {event.step}: loss={event.loss}, grad_norm={event.grad_norm}")
            
            # Run the scenario
            result = driver.run_events(log_events, scenario_name="test_3_max_rollbacks")
            
            print(f"\nResults:")
            print(f"  Total events: {len(result.events)}")
            print(f"  Guard alerts: {len(result.guard_alerts)}")
            print(f"  Guard metrics: {len(result.guard_metrics)}")
            
            # === ASSERTION 1: GUARD_ROLLBACK once ===
            print(f"\n1. Checking GUARD_ROLLBACK count...")
            rollback_alerts = [e for e in result.guard_alerts if e.get("code") == "GUARD_ROLLBACK"]
            self.assertEqual(len(rollback_alerts), 1, 
                           f"Expected 1 GUARD_ROLLBACK, got {len(rollback_alerts)}")
            
            rollback_alert = rollback_alerts[0]
            rollback_step = rollback_alert.get("step") or rollback_alert.get("metadata", {}).get("step")
            print(f"   ✓ Found exactly 1 GUARD_ROLLBACK at step {rollback_step}")
            print(f"   ✓ Message: {rollback_alert.get('message')}")
            
            # === ASSERTION 2: GUARD_ABORT second time with reason max rollbacks ===
            print(f"\n2. Checking GUARD_ABORT with max rollbacks reason...")
            abort_alert = result.assert_alert_present("GUARD_ABORT")
            abort_step = abort_alert.get("step") or abort_alert.get("metadata", {}).get("step")
            self.assertEqual(abort_step, 11, f"Expected GUARD_ABORT at step 11, got step {abort_step}")
            print(f"   ✓ GUARD_ABORT found at step {abort_step}")
            
            # Check message contains max rollbacks context
            abort_message = abort_alert.get("message", "").lower()
            self.assertIn("max rollbacks", abort_message, 
                         f"Expected message to contain 'max rollbacks', got: {abort_alert.get('message')}")
            print(f"   ✓ Message contains max rollbacks: {abort_alert.get('message')}")
            
            # === ASSERTION 3: Verify trigger sequence ===
            print(f"\n3. Checking trigger sequence...")
            triggered_alerts = [e for e in result.guard_alerts if e.get("code") == "GUARD_TRIGGERED"]
            self.assertEqual(len(triggered_alerts), 2, 
                           f"Expected 2 GUARD_TRIGGERED, got {len(triggered_alerts)}")
            
            trigger_steps = result.get_alert_steps("GUARD_TRIGGERED")
            self.assertEqual(set(trigger_steps), {10, 11}, 
                           f"Expected triggers at steps 10,11, got {trigger_steps}")
            print(f"   ✓ Found triggers at steps {trigger_steps}")
            
            # === ASSERTION 4: Verify abort metrics ===
            print(f"\n4. Checking abort metrics...")
            
            # Find abort metrics
            abort_metrics = result.get_metrics_by_action("abort")
            self.assertGreater(len(abort_metrics), 0, "Should have abort metrics")
            
            abort_metric = abort_metrics[0]
            metrics_data = abort_metric.get("metrics", {})
            
            # Verify action is abort
            action = metrics_data.get("action")
            self.assertEqual(action, "abort", f"Expected action 'abort', got '{action}'")
            print(f"   ✓ Found metrics with action=abort")
            
            # Verify rollback count context
            n_rollbacks = metrics_data.get("n_rollbacks")
            if n_rollbacks is not None:
                self.assertGreaterEqual(n_rollbacks, 1, f"Expected n_rollbacks >= 1, got {n_rollbacks}")
                print(f"   ✓ n_rollbacks={n_rollbacks}")
            
            # Verify kind is guard
            kind = abort_metric.get("kind")
            self.assertEqual(kind, "guard", f"Expected kind 'guard', got '{kind}'")
            print(f"   ✓ Metrics kind=guard")
            
            # Show complete abort metrics for debugging
            print(f"\nAbort metrics details:")
            for key, value in metrics_data.items():
                print(f"  {key}: {value}")
            
            # === ASSERTION 5: Verify window behavior ===
            print(f"\n5. Checking window behavior...")
            if hasattr(driver.callback, 'guard') and driver.callback.guard:
                guard = driver.callback.guard
                rollback_count = guard.n_rollbacks
                print(f"   ✓ Total rollbacks in guard: {rollback_count}")
                self.assertEqual(rollback_count, 1, f"Expected 1 rollback, got {rollback_count}")
            
            # === SUMMARY ===
            print(f"\n" + "="*60)
            print("✅ TEST 3 PASSED!")
            print("="*60)
            print(f"Summary:")
            print(f"  • First trigger at step 10 successfully rolled back")
            print(f"  • Second trigger at step 11 aborted (max_rollbacks=1 in window)")
            print(f"  • Window-based anti-thrash protection working correctly")
            print(f"  • No cooldown interference (cooldown_steps=0)")
            print(f"  • JSONL file: {result.telemetry_file}")
            
        finally:
            driver.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)