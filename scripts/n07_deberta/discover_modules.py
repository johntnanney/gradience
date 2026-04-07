#!/usr/bin/env python3
"""
Discover DeBERTa-v3-base module names for LoRA targeting.

Run this FIRST on the GPU box before training to confirm the correct
target_modules list for PEFT LoraConfig.

Usage:
    python scripts/n07_deberta/discover_modules.py

Expected output for DeBERTa-v3-base:
    The four attention projections that map to Q/K/V/O.
    Common names: query_proj, key_proj, value_proj, dense
    (but DeBERTa-v3's disentangled attention may use different names)
"""

from __future__ import annotations


def main():
    from transformers import AutoModel

    print("Loading microsoft/deberta-v3-base...")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

    print("\n=== All named modules with weights ===\n")
    for name, mod in model.named_modules():
        if hasattr(mod, "weight"):
            shape = tuple(mod.weight.shape)
            print(f"  {name:70s} {shape}")

    print("\n=== Attention-related modules ===\n")
    attention_keywords = ["attention", "proj", "query", "key", "value", "dense", "qkv"]
    for name, mod in model.named_modules():
        if any(kw in name.lower() for kw in attention_keywords) and hasattr(mod, "weight"):
            shape = tuple(mod.weight.shape)
            print(f"  {name:70s} {shape}")

    print("\n=== Suggested target_modules for LoRA (attention projections) ===\n")

    # Collect unique leaf module names that look like attention projections
    candidates = set()
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        leaf = name.split(".")[-1]
        # Look for linear layers inside attention blocks
        if any(kw in name.lower() for kw in ["attention", "self"]):
            if len(mod.weight.shape) == 2:  # Linear layer
                candidates.add(leaf)

    print(f"  Candidate leaf names: {sorted(candidates)}")
    print()
    print("  Update LORA_CONFIG['target_modules'] in train_deberta.py if these")
    print("  differ from the default ['query_proj', 'key_proj', 'value_proj', 'dense']")

    # Also check total parameter count for the protocol
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total model parameters: {n_params:,}")


if __name__ == "__main__":
    main()
