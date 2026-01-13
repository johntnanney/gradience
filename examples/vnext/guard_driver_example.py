#!/usr/bin/env python
"""
Example usage of the reusable HF callback driver for Guard testing.

This demonstrates how to use the CPU-friendly driver to test Guard scenarios
without HuggingFace Trainer overhead.

Usage:
    python examples/vnext/guard_driver_example.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gradience.vnext.integrations.hf import GradienceCallback
from tests.helpers.hf_callback_driver import (
    HFCallbackDriver, 
    LogEvent,
    create_grad_explosion_scenario,
    create_cooldown_scenario,
    create_max_rollbacks_scenario
)


def demo_basic_usage():
    """Demonstrate basic driver usage with manual events."""
    
    print("=" * 60)
    print("Demo 1: Basic Driver Usage")
    print("=" * 60)
    
    # Create driver with Guard configuration
    guard_config = {
        "enable_guard": True,
        "guard_snapshot_every": 5,
        "guard_grad_threshold": 100.0,
        "guard_cooldown_steps": 0,
        "guard_max_rollbacks": 3,
    }
    
    driver = HFCallbackDriver(
        callback_class=GradienceCallback,
        callback_config=guard_config
    )
    
    # Define log events manually
    log_events = [
        LogEvent(step=1, loss=2.5, grad_norm=1.0),
        LogEvent(step=2, loss=2.4, grad_norm=1.2),
        LogEvent(step=5, loss=2.3, grad_norm=0.8),   # Should trigger snapshot
        LogEvent(step=10, loss=2.1, grad_norm=500.0),  # Should trigger rollback
    ]
    
    # Run the scenario
    result = driver.run_events(log_events, scenario_name="basic_test")
    
    # Make assertions
    result.assert_alert_present("GUARD_INIT")
    result.assert_alert_present("GUARD_TRIGGERED")
    result.assert_alert_present("GUARD_ROLLBACK")
    result.assert_rollback_count(1)
    
    result.print_summary()
    driver.cleanup()
    
    print("✅ Basic usage demo passed!\n")


def demo_predefined_scenarios():
    """Demonstrate using predefined scenarios."""
    
    print("=" * 60)
    print("Demo 2: Predefined Scenarios")
    print("=" * 60)
    
    # Test gradient explosion scenario
    print("\n--- Gradient Explosion Scenario ---")
    scenario = create_grad_explosion_scenario(trigger_step=8, grad_norm=750.0)
    
    driver = HFCallbackDriver(
        callback_class=GradienceCallback,
        callback_config=scenario.guard_config
    )
    
    result = driver.run_scenario(scenario)
    
    # Verify expected alerts
    for alert_code in scenario.expected_alerts:
        result.assert_alert_present(alert_code)
    
    result.assert_rollback_count(scenario.expected_rollbacks)
    result.print_summary()
    driver.cleanup()
    
    print("✅ Gradient explosion scenario passed!")
    
    # Test cooldown scenario
    print("\n--- Cooldown Lockout Scenario ---")
    cooldown_scenario = create_cooldown_scenario(
        first_step=10, 
        second_step=15, 
        cooldown_steps=20
    )
    
    driver = HFCallbackDriver(
        callback_class=GradienceCallback,
        callback_config=cooldown_scenario.guard_config
    )
    
    result = driver.run_scenario(cooldown_scenario)
    
    # Verify sequence of events
    result.assert_sequence(["GUARD_INIT", "GUARD_TRIGGERED", "GUARD_ROLLBACK", "GUARD_TRIGGERED", "GUARD_ABORT"])
    result.assert_rollback_count(1)  # Only first trigger should rollback
    
    result.print_summary()
    driver.cleanup()
    
    print("✅ Cooldown lockout scenario passed!")


def demo_custom_scenario():
    """Demonstrate creating a custom scenario."""
    
    print("=" * 60)
    print("Demo 3: Custom Scenario")
    print("=" * 60)
    
    # Custom scenario: multiple explosions with snapshots
    guard_config = {
        "enable_guard": True,
        "guard_snapshot_every": 3,
        "guard_grad_threshold": 50.0,  # Lower threshold
        "guard_cooldown_steps": 5,     # Short cooldown
        "guard_max_rollbacks": 2,      # Allow 2 rollbacks
        "guard_window_steps": 20,
    }
    
    # Complex event sequence
    log_events = [
        # Normal training
        LogEvent(step=1, loss=3.0, grad_norm=2.0),
        LogEvent(step=2, loss=2.8, grad_norm=1.5),
        LogEvent(step=3, loss=2.6, grad_norm=3.0),  # Snapshot at step 3
        LogEvent(step=4, loss=2.4, grad_norm=2.0),
        LogEvent(step=5, loss=2.2, grad_norm=1.0),
        
        # First explosion
        LogEvent(step=6, loss=2.0, grad_norm=100.0),  # Trigger + rollback
        
        # Recovery  
        LogEvent(step=7, loss=2.1, grad_norm=1.5),
        LogEvent(step=8, loss=2.0, grad_norm=2.0),
        LogEvent(step=9, loss=1.9, grad_norm=1.0),  # Snapshot at step 9
        
        # Second explosion (outside cooldown)
        LogEvent(step=12, loss=1.8, grad_norm=80.0),  # Another trigger + rollback
        
        # Third explosion (should hit max_rollbacks)
        LogEvent(step=15, loss=1.7, grad_norm=120.0),  # Should abort
    ]
    
    driver = HFCallbackDriver(
        callback_class=GradienceCallback,
        callback_config=guard_config
    )
    
    result = driver.run_events(log_events, scenario_name="custom_multiple_explosions")
    
    # Detailed assertions
    result.assert_alert_present("GUARD_INIT")
    
    # Should have 3 triggers but only 2 rollbacks (3rd hits max_rollbacks)
    triggered_steps = result.get_alert_steps("GUARD_TRIGGERED")
    rollback_steps = result.get_alert_steps("GUARD_ROLLBACK")
    abort_steps = result.get_alert_steps("GUARD_ABORT")
    
    print(f"\nTriggered at steps: {triggered_steps}")
    print(f"Rollbacks at steps: {rollback_steps}")
    print(f"Aborts at steps: {abort_steps}")
    
    assert len(triggered_steps) == 3, f"Expected 3 triggers, got {len(triggered_steps)}"
    assert len(rollback_steps) == 2, f"Expected 2 rollbacks, got {len(rollback_steps)}"
    assert len(abort_steps) == 1, f"Expected 1 abort, got {len(abort_steps)}"
    
    result.assert_rollback_count(2)
    result.print_summary()
    driver.cleanup()
    
    print("✅ Custom scenario passed!")


def main():
    """Run all demo scenarios."""
    
    print("HF Callback Driver Demo")
    print("Testing Guard scenarios with reusable CPU driver")
    
    try:
        demo_basic_usage()
        demo_predefined_scenarios()  
        demo_custom_scenario()
        
        print("=" * 60)
        print("🎉 ALL DEMOS PASSED!")
        print("=" * 60)
        print("\nThe reusable HF callback driver successfully:")
        print("  • Eliminated HuggingFace Trainer overhead")
        print("  • Provided clean assertion helpers")
        print("  • Supported both manual and predefined scenarios")
        print("  • Enabled fast, deterministic Guard testing")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()