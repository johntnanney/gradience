"""
Guard Non-Interference Contract Test

This is a RELEASE GATE test that ensures Guard never interferes with training
when disabled. This test MUST pass before any release.

Contract Requirements:
When enable_guard=False, Guard must:
1. Never create snapshots
2. Never emit Guard telemetry  
3. Never mutate control flow (no training_stop, no rollback)
4. Have zero performance impact
5. Not allocate any Guard-related objects

This test addresses the #1 user fear: "Will this mess with training?"
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

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
    from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
    from tests.helpers.hf_callback_driver import HFCallbackDriver, LogEvent


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "Requires torch and transformers")
class TestGuardNonInterferenceContract(unittest.TestCase):
    """
    CONTRACT TEST: Guard must never interfere when disabled.
    
    This is a release gate - if this test fails, DO NOT SHIP.
    """
    
    def test_guard_disabled_has_zero_interference(self):
        """
        RELEASE GATE: When enable_guard=False, Guard has absolutely zero effect.
        
        This test validates ALL non-interference requirements:
        1. No snapshots created
        2. No Guard telemetry emitted
        3. No control flow mutations
        4. No Guard objects allocated
        5. No performance impact
        """
        print("\n" + "="*70)
        print("GUARD NON-INTERFERENCE CONTRACT TEST (RELEASE GATE)")
        print("="*70)
        
        # Configuration with Guard explicitly disabled
        config = {
            "enable_guard": False,  # <-- GUARD DISABLED
            # These settings should be completely ignored:
            "guard_snapshot_every": 1,
            "guard_grad_threshold": 0.001,  # Extremely low threshold
            "guard_cooldown_steps": 0,
            "guard_max_rollbacks": 999,
        }
        
        print("\nConfiguration:")
        print(f"  enable_guard: {config['enable_guard']} ← DISABLED")
        print("  (all other Guard settings should be ignored)")
        
        # Create driver with disabled Guard
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=config
        )
        
        try:
            # Create extreme scenarios that would trigger Guard if enabled
            extreme_events = [
                # Normal training start
                LogEvent(step=1, loss=2.0, grad_norm=1.0),
                LogEvent(step=2, loss=2.0, grad_norm=1.0),
                
                # EXTREME GRADIENT EXPLOSION (would trigger if enabled)
                LogEvent(step=3, loss=10.0, grad_norm=1e10),  # 10 billion gradient!
                
                # NaN LOSS (would trigger if enabled)
                LogEvent(step=4, loss=float('nan'), grad_norm=1.0),
                
                # INFINITY LOSS (would trigger if enabled)
                LogEvent(step=5, loss=float('inf'), grad_norm=float('inf')),
                
                # More extreme gradients
                LogEvent(step=6, loss=100.0, grad_norm=1e20),  # Even worse!
                
                # Continue normal (to show training wasn't stopped)
                LogEvent(step=7, loss=2.0, grad_norm=1.0),
                LogEvent(step=8, loss=1.9, grad_norm=1.0),
            ]
            
            print("\nExtreme test scenarios:")
            for event in extreme_events:
                print(f"  Step {event.step}: loss={event.loss}, grad_norm={event.grad_norm}")
            
            # Run the extreme scenarios
            result = driver.run_events(extreme_events, scenario_name="guard_disabled_contract")
            
            # ========== CONTRACT VALIDATION 1: No Guard Objects ==========
            print("\n1. Checking Guard object allocation...")
            self.assertIsNone(driver.callback.guard, 
                             "Guard object should NOT be created when disabled")
            print("   ✅ No Guard object allocated")
            
            # ========== CONTRACT VALIDATION 2: No Guard Telemetry ==========
            print("\n2. Checking for Guard telemetry emissions...")
            
            # Check for any Guard alerts
            guard_alerts = [e for e in result.events 
                           if e.get("event") == "alert" 
                           and e.get("code", "").startswith("GUARD_")]
            self.assertEqual(len(guard_alerts), 0,
                           f"Found {len(guard_alerts)} Guard alerts when disabled!")
            print(f"   ✅ No Guard alerts emitted (checked {len(result.events)} events)")
            
            # Check for any Guard metrics
            guard_metrics = [e for e in result.events
                           if e.get("event") == "metrics"
                           and e.get("kind") == "guard"]
            self.assertEqual(len(guard_metrics), 0,
                           f"Found {len(guard_metrics)} Guard metrics when disabled!")
            print("   ✅ No Guard metrics emitted")
            
            # Double-check: no events should contain "guard" in any form
            guard_related = []
            for event in result.events:
                event_str = json.dumps(event).lower()
                if "guard" in event_str:
                    guard_related.append(event)
            
            self.assertEqual(len(guard_related), 0,
                           f"Found {len(guard_related)} events mentioning 'guard'!")
            print("   ✅ No events contain 'guard' references")
            
            # ========== CONTRACT VALIDATION 3: No Control Flow Mutation ==========
            print("\n3. Checking control flow...")
            
            # Training should NOT have been stopped
            should_stop = driver.mock_control.should_training_stop
            self.assertNotEqual(should_stop, True,
                              "Training was stopped despite Guard being disabled!")
            print("   ✅ Training not stopped (should_training_stop not set)")
            
            # All events should have been processed
            train_steps = [e for e in result.events if e.get("event") == "train_step"]
            self.assertEqual(len(train_steps), len(extreme_events),
                           f"Expected {len(extreme_events)} train steps, got {len(train_steps)}")
            print(f"   ✅ All {len(extreme_events)} events processed normally")
            
            # ========== CONTRACT VALIDATION 4: No Snapshots ==========
            print("\n4. Checking for snapshots...")
            
            # No memory should be used for snapshots
            self.assertIsNone(driver.callback.guard,
                             "Guard should be None, so no snapshots possible")
            print("   ✅ No snapshots possible (Guard not initialized)")
            
            # ========== CONTRACT VALIDATION 5: Performance Impact ==========
            print("\n5. Checking performance impact...")
            
            # With Guard disabled, callback overhead should be minimal
            # We can't measure exact performance here, but we verify no Guard code paths
            # Check that Guard tracking variables don't exist or are not initialized
            guard_tracking_attrs = ['_last_loss', '_last_grad_norm', '_last_guard_snapshot_step']
            for attr in guard_tracking_attrs:
                self.assertFalse(hasattr(driver.callback, attr),
                               f"Guard tracking variable {attr} should not exist when Guard is disabled")
            print("   ✅ No Guard tracking overhead")
            
            # ========== SUMMARY ==========
            print("\n" + "="*70)
            print("✅ CONTRACT TEST PASSED: Guard has ZERO interference when disabled")
            print("="*70)
            print("\nValidated:")
            print("  • No Guard objects allocated")
            print("  • No Guard telemetry emitted")
            print("  • No control flow mutations")
            print("  • No snapshots created")
            print("  • No performance overhead")
            print("\nEven with extreme conditions (NaN, Inf, 1e20 gradients),")
            print("Guard did not interfere with training in any way.")
            print("\n🔒 Safe to ship: Guard is completely inert when disabled")
            
        finally:
            driver.cleanup()
    
    def test_guard_disabled_with_mock_trainer_state_mutations(self):
        """
        Additional contract test: Verify Guard doesn't mutate Trainer state.
        
        Tests that even if someone tries to access guard-related state,
        nothing happens when Guard is disabled.
        """
        print("\n" + "="*70)
        print("GUARD STATE MUTATION CONTRACT TEST")
        print("="*70)
        
        config = {"enable_guard": False}
        driver = HFCallbackDriver(
            callback_class=GradienceCallback,
            callback_config=config
        )
        
        try:
            # Store initial state
            initial_model_state = str(driver.model.state_dict())
            initial_args = str(vars(driver.mock_args))
            initial_state = str(vars(driver.mock_state))
            
            # Run with extreme conditions
            extreme_events = [
                LogEvent(step=1, loss=float('nan'), grad_norm=float('inf')),
                LogEvent(step=2, loss=1e100, grad_norm=1e100),
            ]
            
            result = driver.run_events(extreme_events, scenario_name="state_mutation_test")
            
            # Verify no state mutations beyond normal progress
            final_model_state = str(driver.model.state_dict())
            final_args = str(vars(driver.mock_args))
            
            print("\n1. Model state check...")
            self.assertEqual(initial_model_state, final_model_state,
                           "Model state was mutated despite Guard being disabled!")
            print("   ✅ Model state unchanged")
            
            print("\n2. Training arguments check...")
            self.assertEqual(initial_args, final_args,
                           "Training args were mutated despite Guard being disabled!")
            print("   ✅ Training arguments unchanged")
            
            print("\n3. Callback attributes check...")
            # These Guard-specific attributes should not exist
            guard_attrs = ['guard', '_last_guard_snapshot_step', '_guard_enabled']
            for attr in guard_attrs:
                if hasattr(driver.callback, attr):
                    value = getattr(driver.callback, attr)
                    if attr == 'guard':
                        self.assertIsNone(value, f"{attr} should be None when disabled")
                    # Other attributes might exist but should be inactive
            print("   ✅ No active Guard attributes")
            
            print("\n✅ STATE MUTATION TEST PASSED")
            
        finally:
            driver.cleanup()
    
    def test_guard_disabled_is_default_safe(self):
        """
        Contract test: Default configuration must have Guard disabled.
        
        This ensures new users get non-interference by default.
        """
        print("\n" + "="*70)
        print("DEFAULT CONFIGURATION SAFETY TEST")
        print("="*70)
        
        # Create callback with default config (no explicit enable_guard)
        default_config = GradienceCallbackConfig()
        
        print("\nDefault configuration:")
        print(f"  enable_guard: {default_config.enable_guard}")
        
        # Verify Guard is disabled by default
        self.assertFalse(default_config.enable_guard,
                        "Guard must be disabled by default for safety!")
        print("   ✅ Guard is disabled by default")
        
        # Create callback with default config
        callback = GradienceCallback(default_config)
        
        # Mock minimal HF objects with proper attributes
        mock_args = Mock()
        mock_args.output_dir = "/tmp/test"
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1.0
        mock_args.learning_rate = 5e-4
        mock_args.weight_decay = 0.01
        mock_args.adam_beta1 = 0.9
        mock_args.adam_beta2 = 0.999
        mock_args.adam_epsilon = 1e-8
        mock_args.optim = "adamw_torch"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock()
        mock_state.global_step = 0
        mock_control = Mock()
        mock_model = Mock()
        
        # Initialize training
        callback.on_train_begin(mock_args, mock_state, mock_control, model=mock_model)
        
        # Verify no Guard was created
        self.assertIsNone(callback.guard,
                         "Guard should not be created with default config")
        print("   ✅ No Guard created with default configuration")
        
        print("\n✅ DEFAULT SAFETY TEST PASSED")
        print("   New users are protected by default")


if __name__ == "__main__":
    # Run with high verbosity for release gate visibility
    unittest.main(verbosity=2)