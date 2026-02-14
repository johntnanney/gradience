#!/usr/bin/env python3
"""
download_adapters.py — Fetch test adapters from HuggingFace Hub.

Downloads only adapter weights (small), not the full base model.
Total download: ~200-500MB depending on adapter sizes.

Run discover_adapters.py first to verify which repos are available,
then update the ADAPTERS dict below as needed.

Usage:
    python scripts/merge_audit_test/download_adapters.py
"""

import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Configuration — update these after running discover_adapters.py
# ---------------------------------------------------------------------------

ADAPTER_DIR = Path("/workspace/merge_test/adapters")

# Adapter pairs for testing. Keys are local directory names.
# Update repo IDs based on what discover_adapters.py finds.
ADAPTERS = {
    # Pair 1: Code × Customer Support (expect LOW overlap)
    "magicoder": "predibase/magicoder",
    "customer_support": "predibase/customer_support",

    # Pair 2: Summarization variants (expect MODERATE overlap)
    # NOTE: Uncomment and update after verifying availability.
    #       Run discover_adapters.py to check what Predibase has.
    # "tldr": "predibase/tldr",
    # "dialogsum": "predibase/dialogsum",

    # Pair 3: NLU/Classification (expect HIGH overlap / REDUNDANT)
    # NOTE: Uncomment and update after verifying availability.
    # "glue_mnli": "predibase/glue_mnli",
    # "glue_qnli": "predibase/glue_qnli",

    # Pair 4: Non-Predibase adapters (robustness test)
    "zephyr_sft": "alignment-handbook/zephyr-7b-sft-lora",
    # "gsm8k": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
}

# Only download adapter files, not bundled model weights
ALLOW_PATTERNS = [
    "adapter_config.json",
    "adapter_model.*",
    "*.safetensors",
    "*.bin",
    "tokenizer_config.json",  # Sometimes needed for inference verification
    "special_tokens_map.json",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_all():
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading test adapters")
    print("=" * 60)

    successes = []
    failures = []

    for name, repo_id in ADAPTERS.items():
        dest = ADAPTER_DIR / name

        if dest.exists() and (dest / "adapter_config.json").exists():
            print(f"\n  [skip] {name} — already downloaded")
            successes.append(name)
            continue

        print(f"\n  Downloading {repo_id} -> {dest}")
        start = time.time()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                allow_patterns=ALLOW_PATTERNS,
            )
            elapsed = time.time() - start
            print(f"  [ok]   {name} downloaded ({elapsed:.1f}s)")
            successes.append(name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append((name, str(e)))

    return successes, failures


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all():
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    for name in ADAPTERS:
        dest = ADAPTER_DIR / name
        config_path = dest / "adapter_config.json"

        if not config_path.exists():
            print(f"\n  [MISSING] {name}: no adapter_config.json")
            continue

        config = json.loads(config_path.read_text())

        print(f"\n  [ok] {name}")
        print(f"       Base model:     {config.get('base_model_name_or_path', 'unknown')}")
        print(f"       Rank (r):       {config.get('r', 'unknown')}")
        print(f"       Alpha:          {config.get('lora_alpha', 'unknown')}")
        print(f"       Target modules: {config.get('target_modules', 'unknown')}")
        print(f"       Task type:      {config.get('task_type', 'unknown')}")

        # Find weight file and report size
        for ext in ["adapter_model.safetensors", "adapter_model.bin"]:
            weight_path = dest / ext
            if weight_path.exists():
                size_mb = weight_path.stat().st_size / (1024 * 1024)
                print(f"       Weights:        {ext} ({size_mb:.1f} MB)")
                break
        else:
            print(f"       Weights:        NOT FOUND")

    # Cross-check: warn if adapters in a pair have different base models
    print("\n" + "-" * 60)
    print("  Base model compatibility check:")
    configs = {}
    for name in ADAPTERS:
        config_path = ADAPTER_DIR / name / "adapter_config.json"
        if config_path.exists():
            c = json.loads(config_path.read_text())
            configs[name] = c

    pairs = [
        ("magicoder", "customer_support", "Pair 1: Code x Support"),
        ("zephyr_sft", "gsm8k", "Pair 4: Chat x Math"),
    ]

    for a, b, label in pairs:
        if a not in configs or b not in configs:
            print(f"  {label}: skipped (adapter not downloaded)")
            continue

        base_a = configs[a].get("base_model_name_or_path", "?")
        base_b = configs[b].get("base_model_name_or_path", "?")
        rank_a = configs[a].get("r", "?")
        rank_b = configs[b].get("r", "?")
        modules_a = set(configs[a].get("target_modules", []))
        modules_b = set(configs[b].get("target_modules", []))

        if base_a == base_b:
            print(f"  {label}: SAME base model ({base_a})")
        else:
            print(f"  {label}: DIFFERENT base models!")
            print(f"    A: {base_a}")
            print(f"    B: {base_b}")

        if rank_a != rank_b:
            print(f"    WARNING: Different ranks (A={rank_a}, B={rank_b})")

        if modules_a != modules_b:
            shared = modules_a & modules_b
            only_a = modules_a - modules_b
            only_b = modules_b - modules_a
            print(f"    WARNING: Different target modules")
            print(f"      Shared: {shared}")
            if only_a:
                print(f"      Only A: {only_a}")
            if only_b:
                print(f"      Only B: {only_b}")


if __name__ == "__main__":
    successes, failures = download_all()
    verify_all()

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE — {len(successes)} succeeded, {len(failures)} failed")
        for name, err in failures:
            print(f"  FAILED: {name} — {err}")
        print("\nUpdate ADAPTERS dict and re-run, or find alternatives.")
    else:
        print(f"DONE — {len(successes)} adapters ready")
        print("\nNext: python scripts/merge_audit_test/run_merge_audits.py")
    print("=" * 60)
