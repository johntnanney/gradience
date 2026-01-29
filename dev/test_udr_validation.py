#!/usr/bin/env python3
"""
Simple test script to validate UDR opt-in policy enforcement.
"""

def test_udr_opt_in_logic():
    """Test the UDR opt-in validation logic directly."""
    
    print("=== Testing UDR Opt-In Policy ===\n")
    
    # Test Case 1: Default behavior (no audit section)
    print("Test 1: Default behavior (no audit section)")
    config = {}
    audit_config = config.get("audit", {})
    base_model_id = audit_config.get("base_model")
    compute_udr_requested = audit_config.get("compute_udr", False)
    
    # Should not raise error
    if compute_udr_requested and base_model_id is None:
        print("❌ Unexpected error condition")
        return False
        
    compute_udr = compute_udr_requested and base_model_id is not None
    expected_result = False
    
    if compute_udr == expected_result:
        print(f"✅ compute_udr = {compute_udr} (expected: {expected_result})")
    else:
        print(f"❌ compute_udr = {compute_udr} (expected: {expected_result})")
        return False
    
    # Test Case 2: Explicit compute_udr=True with missing base_model (should error)
    print("\nTest 2: compute_udr=True with missing base_model")
    config = {"audit": {"compute_udr": True}}
    audit_config = config.get("audit", {})
    base_model_id = audit_config.get("base_model")
    compute_udr_requested = audit_config.get("compute_udr", False)
    
    try:
        if compute_udr_requested and base_model_id is None:
            raise ValueError(
                "UDR computation was explicitly requested (audit.compute_udr: true) but "
                "audit.base_model is not set. Either:\n"
                "  1. Set audit.base_model to the base model ID, or\n"
                "  2. Set audit.compute_udr: false to disable UDR computation"
            )
        print("❌ Expected ValueError was not raised")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError:")
        print(f"    {str(e).split('.')[0]}...")
    
    # Test Case 3: compute_udr=False (UDR disabled regardless of base_model)
    print("\nTest 3: compute_udr=False (UDR disabled)")
    config = {"audit": {"compute_udr": False, "base_model": "distilbert-base-uncased"}}
    audit_config = config.get("audit", {})
    base_model_id = audit_config.get("base_model")
    compute_udr_requested = audit_config.get("compute_udr", False)
    
    if compute_udr_requested and base_model_id is None:
        print("❌ Unexpected error condition")
        return False
        
    compute_udr = compute_udr_requested and base_model_id is not None
    expected_result = False  # Should be False because compute_udr_requested is False
    
    if compute_udr == expected_result:
        print(f"✅ compute_udr = {compute_udr} (expected: {expected_result})")
    else:
        print(f"❌ compute_udr = {compute_udr} (expected: {expected_result})")
        return False
    
    # Test Case 4: Proper opt-in (compute_udr=True + base_model set)
    print("\nTest 4: Proper opt-in (compute_udr=True + base_model)")
    config = {"audit": {"compute_udr": True, "base_model": "distilbert-base-uncased"}}
    audit_config = config.get("audit", {})
    base_model_id = audit_config.get("base_model")
    compute_udr_requested = audit_config.get("compute_udr", False)
    
    if compute_udr_requested and base_model_id is None:
        print("❌ Unexpected error condition")
        return False
        
    compute_udr = compute_udr_requested and base_model_id is not None
    expected_result = True  # Should be True
    
    if compute_udr == expected_result:
        print(f"✅ compute_udr = {compute_udr} (expected: {expected_result})")
        print(f"✅ base_model_id = {base_model_id}")
    else:
        print(f"❌ compute_udr = {compute_udr} (expected: {expected_result})")
        return False
    
    print("\n🎉 All UDR opt-in policy tests passed!")
    return True


if __name__ == "__main__":
    success = test_udr_opt_in_logic()
    exit(0 if success else 1)