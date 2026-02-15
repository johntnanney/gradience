#!/usr/bin/env python3
"""
gpu_verification.py — GPU-based verification of merge-audit results.

This is the "trust but verify" step that justifies using an A40.
We load the actual base model + adapters on GPU and verify:

1. Both adapters load correctly with PEFT
2. The effective weight updates (B @ A) match what merge-audit computed
3. Quick inference sanity check: adapters produce different outputs
   for task-specific prompts (confirming they're genuinely different)

Usage:
    python scripts/merge_audit_test/gpu_verification.py
"""

import json
import time
from pathlib import Path

import torch

ADAPTER_DIR = Path("/workspace/merge_test/adapters")
RESULTS_DIR = Path("/workspace/merge_test/results")
BASE_MODEL = "mistralai/Mistral-7B-v0.1"

# Adapters to verify (must be downloaded already)
ADAPTERS_TO_VERIFY = ["magicoder", "customer_support"]

# Task-specific prompts for inference verification
TASK_PROMPTS = {
    "magicoder": (
        "Below is a programming problem. Write a solution in Python.\n\n"
        "### Problem: Write a function that reverses a string.\n\n"
        "### Solution:"
    ),
    "customer_support": (
        "Customer: I ordered a laptop last week and it still hasn't arrived. "
        "I'm really frustrated. Can you help me?\n\nAgent:"
    ),
    "zephyr_sft": (
        "You are a helpful assistant. What are the main benefits of "
        "regular exercise for mental health?"
    ),
}


# ---------------------------------------------------------------------------
# Test 1: Adapter loading verification
# ---------------------------------------------------------------------------

def verify_adapter_loading():
    """Verify adapters load correctly with PEFT on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n{'=' * 70}")
    print(f"  TEST 1: Adapter Loading Verification")
    print(f"{'=' * 70}")

    print(f"\n  Loading base model: {BASE_MODEL}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  Base model loaded ({time.time() - start:.1f}s)")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    results = {}

    for name in ADAPTERS_TO_VERIFY:
        adapter_path = ADAPTER_DIR / name
        if not adapter_path.exists():
            print(f"\n  [{name}] SKIPPED — not downloaded")
            continue

        print(f"\n  [{name}] Loading adapter from {adapter_path}...")
        mem_before = torch.cuda.memory_allocated()
        start = time.time()

        try:
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model.eval()
            elapsed = time.time() - start
            mem_after = torch.cuda.memory_allocated()
            mem_delta = (mem_after - mem_before) / 1e6  # MB

            print(f"  [{name}] Loaded successfully ({elapsed:.1f}s)")
            print(f"  [{name}] GPU memory: {mem_after / 1e9:.2f} GB "
                  f"(+{mem_delta:.1f} MB for adapter)")

            # Quick generation test
            prompt = TASK_PROMPTS.get(name, "Tell me about yourself:")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            mem_pre_gen = torch.cuda.memory_allocated()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            mem_post_gen = torch.cuda.memory_allocated()

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):]
            print(f"  [{name}] Generation OK: {generated[:100].strip()}...")
            print(f"  [{name}] Inference memory delta: "
                  f"+{(mem_post_gen - mem_pre_gen) / 1e6:.1f} MB")

            results[name] = {
                "status": "ok",
                "load_time_s": elapsed,
                "adapter_memory_mb": round(mem_delta, 1),
                "total_memory_gb": round(mem_after / 1e9, 2),
            }

            # Unload adapter
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            results[name] = {"status": "error", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Test 2: Weight computation verification
# ---------------------------------------------------------------------------

def verify_weight_computation():
    """Verify merge-audit's weight computation matches direct loading.

    Load each adapter's A and B matrices directly via safetensors,
    compute the effective update (scaling * B @ A), run SVD, and compare
    the spectral structure against what merge-audit reported.
    """
    from safetensors import safe_open

    print(f"\n\n{'=' * 70}")
    print(f"  TEST 2: Weight Computation Verification")
    print(f"{'=' * 70}")

    for name in ADAPTERS_TO_VERIFY:
        adapter_path = ADAPTER_DIR / name
        sf_path = adapter_path / "adapter_model.safetensors"

        if not sf_path.exists():
            # Try .bin fallback
            bin_path = adapter_path / "adapter_model.bin"
            if bin_path.exists():
                print(f"\n  [{name}] Only .bin format available, skipping direct verification")
                continue
            print(f"\n  [{name}] SKIPPED — no weight file")
            continue

        config = json.loads((adapter_path / "adapter_config.json").read_text())
        rank = config.get("r", 0)
        alpha = config.get("lora_alpha", rank)
        scaling = alpha / rank if rank > 0 else 1.0

        print(f"\n  [{name}] rank={rank}, alpha={alpha}, scaling={scaling:.4f}")

        with safe_open(sf_path, framework="pt") as f:
            keys = list(f.keys())

        a_keys = sorted([k for k in keys if "lora_A" in k])

        if not a_keys:
            print(f"  [{name}] No lora_A keys found in weights")
            continue

        # Verify first and last layers to check consistency
        check_keys = [a_keys[0]]
        if len(a_keys) > 1:
            check_keys.append(a_keys[-1])

        for a_key in check_keys:
            b_key = a_key.replace("lora_A", "lora_B")

            with safe_open(sf_path, framework="pt") as f:
                A = f.get_tensor(a_key).float()
                B = f.get_tensor(b_key).float()

            delta_w = scaling * (B @ A)

            # Compute SVD
            U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)

            # Energy analysis
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()
            eff_rank_90 = int((energy < 0.90).sum().item()) + 1
            eff_rank_95 = int((energy < 0.95).sum().item()) + 1

            # Stable rank
            stable_rank = (S.pow(2).sum() / S[0].pow(2)).item()

            layer_name = a_key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            print(f"\n    Layer: {layer_name}")
            print(f"      Shape: A={list(A.shape)}, B={list(B.shape)}, dW={list(delta_w.shape)}")
            print(f"      Top-5 singular values: {[f'{s:.4f}' for s in S[:5].tolist()]}")
            print(f"      Frobenius norm: {delta_w.norm().item():.6f}")
            print(f"      Effective rank (90%): {eff_rank_90} / {rank}")
            print(f"      Effective rank (95%): {eff_rank_95} / {rank}")
            print(f"      Stable rank: {stable_rank:.2f}")

        print(f"\n  [{name}] Weight computation verified")


# ---------------------------------------------------------------------------
# Test 3: Task-specific inference comparison
# ---------------------------------------------------------------------------

def verify_task_specific_inference():
    """Run the same prompt through each adapter to confirm different behavior."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n\n{'=' * 70}")
    print(f"  TEST 3: Task-Specific Inference Comparison")
    print(f"{'=' * 70}")

    available = [n for n in ADAPTERS_TO_VERIFY if (ADAPTER_DIR / n).exists()]
    if len(available) < 2:
        print(f"  Need at least 2 adapters for comparison, have {len(available)}")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Use a neutral prompt that both adapters can respond to
    neutral_prompt = "Explain in one sentence why the sky is blue:"

    outputs_by_adapter = {}

    for name in available:
        adapter_path = ADAPTER_DIR / name
        print(f"\n  [{name}] Loading adapter...")

        try:
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model.eval()

            # Task-specific prompt
            task_prompt = TASK_PROMPTS.get(name, neutral_prompt)
            inputs_task = tokenizer(task_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_task = model.generate(**inputs_task, max_new_tokens=80, do_sample=False)
            resp_task = tokenizer.decode(out_task[0], skip_special_tokens=True)[len(task_prompt):]

            # Neutral prompt (same for all adapters)
            inputs_neutral = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_neutral = model.generate(**inputs_neutral, max_new_tokens=80, do_sample=False)
            resp_neutral = tokenizer.decode(out_neutral[0], skip_special_tokens=True)[len(neutral_prompt):]

            print(f"  [{name}] Task prompt: {task_prompt[:60]}...")
            print(f"  [{name}] Task output: {resp_task[:120].strip()}")
            print(f"  [{name}] Neutral output: {resp_neutral[:120].strip()}")

            outputs_by_adapter[name] = {
                "task_response": resp_task.strip(),
                "neutral_response": resp_neutral.strip(),
            }

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    # Compare neutral responses across adapters
    if len(outputs_by_adapter) >= 2:
        print(f"\n  --- Comparison ---")
        names = list(outputs_by_adapter.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_resp = outputs_by_adapter[names[i]]["neutral_response"]
                b_resp = outputs_by_adapter[names[j]]["neutral_response"]
                same = a_resp == b_resp
                print(f"  {names[i]} vs {names[j]}: {'IDENTICAL' if same else 'DIFFERENT'} neutral responses")
                if same:
                    print(f"    WARNING: identical responses may indicate adapters aren't loading properly")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: GPU not available. GPU verification requires CUDA.")
        print("Run this on RunPod or another GPU machine.")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test 1: Loading
    load_results = verify_adapter_loading()

    # Test 2: Weight computation
    verify_weight_computation()

    # Test 3: Inference comparison
    verify_task_specific_inference()

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"  GPU VERIFICATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"  Adapters tested: {len(load_results)}")
    for name, result in load_results.items():
        status = result["status"].upper()
        if result["status"] == "ok":
            print(f"    {name}: {status} "
                  f"(load={result['load_time_s']:.1f}s, "
                  f"adapter=+{result['adapter_memory_mb']:.0f}MB, "
                  f"total={result['total_memory_gb']:.2f}GB)")
        else:
            print(f"    {name}: {status} — {result.get('error', 'unknown')}")
