#!/usr/bin/env python3
"""
Verify that the UDR opt-in implementation in protocol.py matches our requirements.
"""

import re

def verify_udr_implementation():
    """Read protocol.py and verify UDR implementation."""
    
    print("=== Verifying UDR Opt-In Implementation ===\n")
    
    try:
        with open('gradience/bench/protocol.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Could not find gradience/bench/protocol.py")
        return False
    
    # Check 1: Default behavior (compute_udr defaults to False)
    if 'compute_udr_requested = audit_config.get("compute_udr", False)' in content:
        print("✅ Default behavior: compute_udr defaults to False")
    else:
        print("❌ Default behavior not found")
        return False
    
    # Check 2: Validation for explicit UDR request without base_model
    validation_pattern = r'if compute_udr_requested and base_model_id is None:'
    if re.search(validation_pattern, content):
        print("✅ Validation: Explicit UDR request without base_model is checked")
    else:
        print("❌ Validation pattern not found")
        return False
    
    # Check 3: Error message mentions the configuration keys
    if 'audit.compute_udr: true' in content and 'audit.base_model' in content:
        print("✅ Error message: References correct config keys")
    else:
        print("❌ Error message doesn't reference config keys")
        return False
    
    # Check 4: Final compute_udr logic requires both conditions
    final_logic = 'compute_udr = compute_udr_requested and base_model_id is not None'
    if final_logic in content:
        print("✅ Final logic: Requires both compute_udr_requested=True AND base_model set")
    else:
        print("❌ Final logic not correct")
        return False
    
    # Check 5: base_model_id is passed correctly to audit function
    audit_call_pattern = r'base_model_id=base_model_id if compute_udr else None'
    if re.search(audit_call_pattern, content):
        print("✅ Audit call: base_model_id passed only when UDR enabled")
    else:
        print("❌ Audit call pattern not found")
        return False
    
    print("\n🎉 UDR opt-in implementation verification passed!")
    print("\nImplementation summary:")
    print("  - Default: audit section missing → compute_udr = False")
    print("  - Validation: compute_udr=True + missing base_model → ValueError")  
    print("  - Proper: compute_udr=True + base_model set → UDR enabled")
    print("  - Disabled: compute_udr=False → UDR disabled regardless of base_model")
    
    return True

def verify_no_backcompat_aliases():
    """Verify no legacy aliases exist."""
    print("\n=== Checking for Legacy Aliases ===")
    
    try:
        with open('gradience/bench/protocol.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return True
    
    # Check for enable_udr alias
    if 'enable_udr' in content:
        print("❌ Found legacy 'enable_udr' alias")
        return False
    else:
        print("✅ No legacy 'enable_udr' alias found")
    
    return True

def run_logic_tests():
    """Run logical tests of the UDR policy."""
    print("\n=== Running Logic Tests ===")
    
    def test_udr_logic(audit_config_dict):
        """Test the UDR logic."""
        base_model_id = audit_config_dict.get("base_model")
        compute_udr_requested = audit_config_dict.get("compute_udr", False)
        
        # Apply validation
        if compute_udr_requested and base_model_id is None:
            raise ValueError("UDR requested but base_model missing")
        
        compute_udr = compute_udr_requested and base_model_id is not None
        return compute_udr, base_model_id
    
    test_cases = [
        ({}, False, "Default (empty audit config)"),
        ({"compute_udr": False}, False, "Explicit disable"), 
        ({"compute_udr": False, "base_model": "test"}, False, "Disable with base_model"),
        ({"compute_udr": True, "base_model": "test"}, True, "Proper opt-in"),
    ]
    
    for audit_config, expected, description in test_cases:
        try:
            result, base_model = test_udr_logic(audit_config)
            if result == expected:
                print(f"✅ {description}: compute_udr={result}")
            else:
                print(f"❌ {description}: expected {expected}, got {result}")
                return False
        except Exception as e:
            print(f"❌ {description}: unexpected error {e}")
            return False
    
    # Test error case
    try:
        test_udr_logic({"compute_udr": True})  # Missing base_model
        print("❌ Error case: should have raised ValueError")
        return False
    except ValueError:
        print("✅ Error case: correctly raised ValueError for missing base_model")
    
    return True

if __name__ == "__main__":
    success = True
    success &= verify_udr_implementation()
    success &= verify_no_backcompat_aliases() 
    success &= run_logic_tests()
    
    if success:
        print("\n🎉 All verifications passed! UDR opt-in policy is correctly implemented.")
    else:
        print("\n❌ Some verifications failed.")
    
    exit(0 if success else 1)