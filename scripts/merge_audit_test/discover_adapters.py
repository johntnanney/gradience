#!/usr/bin/env python3
"""
discover_adapters.py — List Predibase LoRA Land adapters on HuggingFace.

Run this first to see what's actually available before downloading.
Searches for Mistral-7B-based LoRA adapters from Predibase and other
sources suitable for merge-audit testing.

Usage:
    python scripts/merge_audit_test/discover_adapters.py
"""

import json

from huggingface_hub import HfApi


def discover_predibase():
    """List all Predibase Mistral-7B LoRA adapters.

    The Predibase collection naming can be inconsistent — some adapters
    are under predibase/ directly, others may be nested differently.
    We cast a wide net with multiple search strategies.
    """
    api = HfApi()

    print("=" * 70)
    print("Predibase LoRA Land — Mistral-7B Adapters")
    print("=" * 70)

    # Strategy 1: author=predibase, search Mistral
    seen = set()
    results = []

    for search_term in ["Mistral", "mistral", "lora", ""]:
        try:
            models = api.list_models(
                author="predibase",
                search=search_term if search_term else None,
                sort="downloads",
                direction=-1,
                limit=100,
            )
            for model in models:
                if model.modelId not in seen:
                    seen.add(model.modelId)
                    results.append(model)
        except Exception as e:
            print(f"  (search '{search_term}' failed: {e})")

    if results:
        print(f"\n  Found {len(results)} Predibase models:")
        for model in sorted(results, key=lambda m: -(m.downloads or 0)):
            print(f"\n  {model.modelId}")
            print(f"    Downloads: {model.downloads}")
            tags = model.tags[:8] if model.tags else []
            print(f"    Tags: {', '.join(tags) if tags else 'none'}")
    else:
        print("\n  No Predibase adapters found via API search.")
        print("  Browse manually: https://huggingface.co/predibase")
        print("  The LoRA Land collection may use a different org name.")

    return results


def discover_specific_candidates():
    """Check availability of known adapter candidates for each pair."""
    api = HfApi()

    print("\n\n" + "=" * 70)
    print("Specific Adapter Candidates — Availability Check")
    print("=" * 70)

    # Organized by pair hypothesis
    candidates = {
        "Pair 1 — Code x Support (LOW overlap)": [
            "predibase/magicoder",
            "predibase/customer_support",
        ],
        "Pair 2 — Summarization variants (MODERATE overlap)": [
            "predibase/tldr",
            "predibase/dialogsum",
            "predibase/samsum",  # fallback: dialogue summarization
            "predibase/xsum",  # fallback: news summarization
        ],
        "Pair 3 — NLU/Classification (HIGH overlap)": [
            "predibase/glue_mnli",
            "predibase/glue_qnli",
            "predibase/glue_sst2",  # fallback: sentiment
            "predibase/glue_qqp",  # fallback: paraphrase detection
        ],
        "Pair 4 — Non-Predibase (ROBUSTNESS)": [
            "alignment-handbook/zephyr-7b-sft-lora",
            "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
        ],
    }

    available = {}

    for pair_label, repos in candidates.items():
        print(f"\n  --- {pair_label} ---")
        for repo_id in repos:
            try:
                info = api.model_info(repo_id)
                print(f"    {repo_id}")
                print(f"      Downloads: {info.downloads}")
                print("      Status: AVAILABLE")

                # Try to peek at adapter config for key details
                try:
                    from huggingface_hub import hf_hub_download

                    config_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="adapter_config.json",
                        cache_dir="/tmp/hf_discover_cache",
                    )
                    config = json.loads(open(config_path).read())
                    base = config.get("base_model_name_or_path", "?")
                    rank = config.get("r", "?")
                    modules = config.get("target_modules", "?")
                    print(f"      Base: {base}")
                    print(f"      Rank: {rank}, Modules: {modules}")
                except Exception:
                    pass  # Config peek is best-effort

                available[repo_id] = True

            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "not found" in err_msg.lower():
                    print(f"    {repo_id}")
                    print("      Status: NOT FOUND")
                else:
                    print(f"    {repo_id}")
                    print(f"      Status: ERROR — {err_msg[:80]}")
                available[repo_id] = False

    return available


def discover_broad_search():
    """Broad search for Mistral-7B LoRA adapters as additional candidates."""
    api = HfApi()

    print("\n\n" + "=" * 70)
    print("Broad Search — Other Mistral-7B LoRA Adapters")
    print("=" * 70)

    search_terms = [
        "Mistral-7B lora",
        "Mistral-7B-v0.1 lora adapter",
        "mistral lora peft",
    ]

    seen = set()
    for term in search_terms:
        try:
            models = api.list_models(
                search=term,
                sort="downloads",
                direction=-1,
                limit=20,
            )
            for model in models:
                if model.modelId not in seen and (model.downloads or 0) > 10:
                    seen.add(model.modelId)
                    print(f"\n  {model.modelId}")
                    print(f"    Downloads: {model.downloads}")
        except Exception:
            pass

    if not seen:
        print("  No additional adapters found.")


if __name__ == "__main__":
    predibase = discover_predibase()
    available = discover_specific_candidates()
    discover_broad_search()

    # Build recommendation
    print("\n\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    available_repos = [r for r, ok in available.items() if ok]
    missing_repos = [r for r, ok in available.items() if not ok]

    if available_repos:
        print(f"\n  Available ({len(available_repos)}):")
        for r in available_repos:
            print(f"    + {r}")

    if missing_repos:
        print(f"\n  Not found ({len(missing_repos)}):")
        for r in missing_repos:
            print(f"    - {r}")
        print("\n  Update download_adapters.py ADAPTERS dict to use available repos.")
        print("  Substitute missing adapters with alternatives from the broad search.")

    print("""
  NEXT STEPS:
  1. Update ADAPTERS dict in download_adapters.py with available repos
  2. Ensure at least 3 pairs covering LOW/MODERATE/HIGH overlap hypotheses
  3. Run: python scripts/merge_audit_test/download_adapters.py
""")
