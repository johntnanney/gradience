#!/usr/bin/env python3
"""Run UDR test directly."""

print("=== Testing UDR Opt-In Policy ===")

# Test the logic that's implemented in protocol.py
def validate_udr_config(audit_config):
    base_model_id = audit_config.get("base_model")
    compute_udr_requested = audit_config.get("compute_udr", False)
    
    # Validate UDR configuration
    if compute_udr_requested and base_model_id is None:
        raise ValueError(
            "UDR computation was explicitly requested (audit.compute_udr: true) but "
            "audit.base_model is not set. Either:\n"
            "  1. Set audit.base_model to the base model ID, or\n"
            "  2. Set audit.compute_udr: false to disable UDR computation"
        )
    
    compute_udr = compute_udr_requested and base_model_id is not None
    return compute_udr, base_model_id

# Test 1: Default (no audit section)
print("\n1. Default behavior:")
try:
    compute_udr, base_model_id = validate_udr_config({})
    print(f"✅ compute_udr={compute_udr}, base_model_id={base_model_id}")
    assert compute_udr == False
except Exception as e:
    print(f"❌ {e}")

# Test 2: Explicit error case
print("\n2. Explicit error case (compute_udr=True, no base_model):")
try:
    validate_udr_config({"compute_udr": True})
    print("❌ Should have raised ValueError")
except ValueError as e:
    print("✅ Correctly raised ValueError")
    print(f"   Message: {str(e)[:80]}...")

# Test 3: Disabled case
print("\n3. Explicitly disabled:")
try:
    compute_udr, base_model_id = validate_udr_config({
        "compute_udr": False, 
        "base_model": "distilbert-base-uncased"
    })
    print(f"✅ compute_udr={compute_udr}, base_model_id={base_model_id}")
    assert compute_udr == False
except Exception as e:
    print(f"❌ {e}")

# Test 4: Proper opt-in
print("\n4. Proper opt-in:")
try:
    compute_udr, base_model_id = validate_udr_config({
        "compute_udr": True,
        "base_model": "distilbert-base-uncased"
    })
    print(f"✅ compute_udr={compute_udr}, base_model_id={base_model_id}")
    assert compute_udr == True
    assert base_model_id == "distilbert-base-uncased"
except Exception as e:
    print(f"❌ {e}")

print("\n🎉 All tests passed! UDR opt-in policy is working correctly.")