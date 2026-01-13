"""
Test Guard functionality using the reusable HF callback driver.

This demonstrates how the driver can be used in actual test files
for clean, fast Guard validation.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

if HAS_TORCH and HAS_TRANSFORMERS:
    from gradience.vnext.integrations.hf import GradienceCallback
    from tests.helpers.hf_callback_driver import (
        HFCallbackDriver, 
        LogEvent,
        create_grad_explosion_scenario,
        create_cooldown_scenario,
        create_max_rollbacks_scenario
    )


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "Requires torch and transformers")
class TestGuardWithDriver(unittest.TestCase):
    """Test Guard functionality using the reusable CPU driver."""
    
    def test_basic_gradient_explosion(self):
        """Test basic gradient explosion detection and rollback."""
        scenario = create_grad_explosion_scenario(trigger_step=10, grad_norm=500.0)
        
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=scenario.guard_config
        )
        
        try:
            result = driver.run_scenario(scenario)
            
            # Verify Guard behavior
            result.assert_alert_present("GUARD_INIT")
            result.assert_alert_present("GUARD_TRIGGERED", "grad_explosion")
            result.assert_alert_present("GUARD_ROLLBACK")
            result.assert_rollback_count(1)
            
            # Verify trigger happened at correct step
            trigger_steps = result.get_alert_steps("GUARD_TRIGGERED")
            self.assertEqual(trigger_steps, [10])
            
        finally:
            driver.cleanup()
    
    def test_cooldown_protection(self):
        """Test cooldown anti-thrash protection."""
        scenario = create_cooldown_scenario(
            first_step=10, 
            second_step=15, 
            cooldown_steps=20
        )
        
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=scenario.guard_config
        )
        
        try:
            result = driver.run_scenario(scenario)
            
            # Verify sequence: first rollback, then abort
            result.assert_sequence([
                "GUARD_INIT", 
                "GUARD_TRIGGERED", 
                "GUARD_ROLLBACK", 
                "GUARD_TRIGGERED", 
                "GUARD_ABORT"
            ])
            
            # Should only have 1 rollback (second was aborted)
            result.assert_rollback_count(1)
            
            # Verify triggers at correct steps
            trigger_steps = result.get_alert_steps("GUARD_TRIGGERED")
            self.assertEqual(set(trigger_steps), {10, 15})
            
        finally:
            driver.cleanup()
    
    def test_max_rollbacks_protection(self):
        """Test max rollbacks anti-thrash protection."""
        scenario = create_max_rollbacks_scenario(
            first_step=10,
            second_step=11, 
            max_rollbacks=1
        )
        
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=scenario.guard_config
        )
        
        try:
            result = driver.run_scenario(scenario)
            
            # Verify abort due to max rollbacks
            result.assert_alert_present("GUARD_ABORT", "max rollbacks")
            result.assert_rollback_count(1)
            
            # Should have 2 triggers but only 1 rollback
            trigger_steps = result.get_alert_steps("GUARD_TRIGGERED")
            rollback_steps = result.get_alert_steps("GUARD_ROLLBACK")
            abort_steps = result.get_alert_steps("GUARD_ABORT")
            
            self.assertEqual(len(trigger_steps), 2)
            self.assertEqual(len(rollback_steps), 1) 
            self.assertEqual(len(abort_steps), 1)
            
        finally:
            driver.cleanup()
    
    def test_disabled_guard(self):
        """Test that disabled Guard doesn't interfere."""
        guard_config = {
            "enable_guard": False,  # Guard disabled
        }
        
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=guard_config
        )
        
        try:
            # Even with exploded gradients, Guard should not activate
            log_events = [
                LogEvent(step=1, loss=2.5, grad_norm=1.0),
                LogEvent(step=2, loss=2.3, grad_norm=1000.0),  # Would trigger if enabled
                LogEvent(step=3, loss=2.1, grad_norm=2000.0),  # Even bigger explosion
            ]
            
            result = driver.run_events(log_events, scenario_name="disabled_guard")
            
            # Should have no Guard alerts (except possibly GUARD_DISABLED)
            guard_alerts = result.guard_alerts
            guard_codes = [alert.get("code") for alert in guard_alerts]
            
            # No GUARD_TRIGGERED or GUARD_ROLLBACK should occur
            self.assertNotIn("GUARD_TRIGGERED", guard_codes)
            self.assertNotIn("GUARD_ROLLBACK", guard_codes)
            
            # Verify callback has no guard instance
            self.assertIsNone(driver.callback.guard)
            
        finally:
            driver.cleanup()
    
    def test_manual_event_construction(self):
        """Test manual event construction for edge cases.""" 
        guard_config = {
            "enable_guard": True,
            "guard_snapshot_every": 2,
            "guard_grad_threshold": 10.0,  # Very low threshold
        }
        
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=guard_config
        )
        
        try:
            # Construct complex scenario manually
            log_events = [
                LogEvent(step=1, loss=5.0, grad_norm=1.0),
                LogEvent(step=2, loss=4.0, grad_norm=2.0),    # Snapshot
                LogEvent(step=3, loss=3.0, grad_norm=15.0),   # Trigger
                LogEvent(step=4, loss=6.0, grad_norm=1.0),    # After rollback
                LogEvent(step=5, loss=2.0, grad_norm=0.5),
            ]
            
            result = driver.run_events(log_events, scenario_name="manual_events")
            
            # Verify basic Guard operation
            result.assert_alert_present("GUARD_TRIGGERED")
            result.assert_rollback_count(1)
            
            # Check snapshot creation
            snapshot_alerts = [a for a in result.guard_alerts if a.get("code") == "GUARD_SNAPSHOT"]
            self.assertGreater(len(snapshot_alerts), 0, "Should have snapshot alerts")
            
        finally:
            driver.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)