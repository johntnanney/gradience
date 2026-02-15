"""
Bench protocol (v0.1).

Implements:
1) Train probe adapter (r=16)
2) Audit -> suggestions
3) Retrain: uniform_median, uniform_p90, per_layer
4) Eval all
5) Emit report (JSON + Markdown)

Step 3.1 implementation: Train probe with GradienceCallback

This module has been decomposed into focused sub-modules for maintainability.
All public symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
#
# Every symbol that was previously importable from gradience.bench.protocol
# is re-exported below so that existing code continues to work unchanged.
# ---------------------------------------------------------------------------

from gradience.bench.model_setup import (  # noqa: F401
    HAS_TRAINING_DEPS,
    load_config,
    setup_dataset,
    setup_model_and_tokenizer,
    setup_compressed_model_and_tokenizer,
    _unwrap_model_for_save,
    _save_peft_adapter_only,
)

from gradience.bench.probe import (  # noqa: F401
    run_probe_audit,
    run_probe_training,
)

from gradience.bench.compression import (  # noqa: F401
    _resolve_policy_rank_source,
    _get_probe_quality_threshold,
    round_to_allowed_ranks,
    _create_shuffled_rank_pattern,
    generate_svd_variant_config,
    get_rank_source_from_config,
    generate_compression_configs,
)

from gradience.bench.variants import (  # noqa: F401
    run_post_tuning,
    run_svd_truncation_variant,
    run_compressed_variant_training,
    run_all_compressed_variants,
)

from gradience.bench.verdicts import (  # noqa: F401
    classify_validation_level,
    compute_verdicts,
)

from gradience.bench.reporting import (  # noqa: F401
    write_probe_eval_json,
    _extract_accuracy_with_fallback,
    gather_environment_info,
    get_git_commit,
    get_git_tag,
    get_hf_model_revision,
    get_dataset_revision,
    extract_model_dataset_info,
    get_primary_metric_key,
    create_config_hash,
    create_canonical_bench_report,
    create_markdown_report,
)

from gradience.bench.multi_seed import (  # noqa: F401
    create_multi_seed_aggregated_report,
    create_multi_seed_markdown_report,
    run_multi_seed_bench_protocol,
)

from gradience.bench.preflight import (  # noqa: F401
    run_bench_preflight_check,
    run_artifact_hygiene_cleanup,
)

from gradience.bench.config_schema import validate_config  # noqa: F401

from gradience.bench.task_profiles import get_task_profile_from_config  # noqa: F401
from gradience.vnext.audit.lora_audit import audit_lora_peft_dir  # noqa: F401

# Imports used by run_bench_protocol itself
from gradience.bench.heartbeat import heartbeat_stage
from gradience.bench.monitored_stage import (
    monitor_generation, setup_global_monitoring
)
from gradience.bench.stage_state import create_stage_manager
from gradience.bench.decision_trace import DecisionTrace


def run_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
    ci: bool = False,
    fast_mode: bool = True,
    max_candidates: int = 4,
    resume: bool = False
) -> Dict[str, Any]:
    """
    Run the complete bench protocol.

    Supports both single-seed and multi-seed configurations.
    Multi-seed configs are detected by presence of 'seeds' in compression config.
    """
    # Set up global monitoring infrastructure
    setup_global_monitoring()

    # Load configuration to check for multi-seed
    config = load_config(config_path)

    # Check for multi-seed configuration
    compression = config.get("compression", {})
    seeds = compression.get("seeds")
    variants_to_test = compression.get("variants_to_test")

    if seeds and len(seeds) > 1:
        print("Detected multi-seed configuration - running aggregated benchmark...")
        return run_multi_seed_bench_protocol(
            config_path=config_path,
            output_dir=output_dir,
            seeds=seeds,
            variants_to_test=variants_to_test,
            smoke=smoke,
            ci=ci
        )

    # Single-seed protocol (original implementation)
    print("Gradience Bench Protocol v0.1")
    print("=" * 40)

    # HYGIENE: Ensure output directory exists BEFORE any logging/tee operations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # HYGIENE: Start heartbeat for single-seed run (prevent SSH timeouts)
    seed = config.get("train", {}).get("seed", 42)
    heartbeat_stage("single_seed_benchmark", output_dir=output_path, seed=seed)

    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Task: {config['task']['dataset']}/{config['task']['subset']}")
    print(f"Smoke mode: {smoke}")
    print()

    # Initialize stage state manager for resume functionality
    stage_manager = create_stage_manager(output_path)

    if resume:
        print("🔄 Resume mode enabled - checking for completed stages...")
        stage_manager.clean_invalid_state()  # Clean up any stale state
        summary = stage_manager.get_resume_summary()
        print(f"   Found {summary['total_completed_stages']} completed stages, {summary['total_completed_variants']} completed variants")
        if summary['total_completed_stages'] > 0 or summary['total_completed_variants'] > 0:
            print("   Completed stages:", ", ".join(summary['completed_stages']) if summary['completed_stages'] else "none")
            print("   Completed variants:", ", ".join(summary['completed_variants']) if summary['completed_variants'] else "none")

    # Preflight checks to catch common failure modes early
    run_bench_preflight_check(config, config['model']['name'])

    # Steps 3.1-3.3: Train, evaluate, and audit probe
    print("Step 3.1-3.3: Training, evaluating, and auditing probe adapter...")
    probe_results = run_probe_training(config_path, output_path, smoke=smoke, stage_manager=stage_manager, resume=resume)

    # Step 3.4: Generate compression configurations
    print("\nStep 3.4: Generating compression configurations...")
    probe_rank = config["lora"]["probe_r"]
    probe_dir = output_path / f"probe_r{probe_rank}"

    # Check probe validity before wasting GPU cycles on compression
    audit_json_path = probe_dir / "audit.json"
    if audit_json_path.exists():
        with open(audit_json_path) as f:
            audit_data = json.load(f)

        probe_validity = audit_data.get("probe_validity", {})
        if not probe_validity.get("valid", True):
            print("")
            print("⚠️" * 15)
            print("⚠️  SKIPPING COMPRESSION VARIANTS")
            print(f"⚠️  Reason: {probe_validity.get('reason', 'UNKNOWN')}")
            print(f"⚠️  Message: {probe_validity.get('message', 'Probe invalid')}")
            print(f"⚠️  stable_rank_mean: {probe_validity.get('stable_rank_mean', 'N/A')}")
            print(f"⚠️  utilization_mean: {probe_validity.get('utilization_mean', 'N/A')}")
            print("⚠️")
            print("⚠️  This prevents wasted GPU cycles on compression variants")
            print("⚠️  when the probe itself is invalid.")
            print("⚠️" * 15)
            print("")

            # Create minimal report for invalid probe and exit early
            verdict_analysis = compute_verdicts(
                probe_results={},
                variant_results={},
                config=config,
                output_path=output_path,
                smoke=smoke
            )

            decision_trace = DecisionTrace(probe_rank=probe_rank)
            canonical_report = create_canonical_bench_report(
                probe_results={},
                variant_results={},
                verdict_analysis=verdict_analysis,
                audit_data=audit_data,
                compression_configs={},
                config=config,
                output_dir=output_path,
                decision_trace=decision_trace
            )

            # Write report files
            report_path = output_path / "bench.json"
            with open(report_path, 'w') as f:
                json.dump(canonical_report, f, indent=2, ensure_ascii=False)

            return canonical_report

    # Check if config generation can be skipped
    skip_config_gen = resume and stage_manager and stage_manager.should_skip_config_generation()

    if not skip_config_gen:
        seed = config.get("train", {}).get("seed", 42)
        with monitor_generation("generate_configs", output_dir=output_path, seed=seed) as stage:
            stage.progress("Analyzing probe audit results")
            compression_configs, decision_trace = generate_compression_configs(probe_dir, config, fast_mode=fast_mode, max_candidates=max_candidates)
            stage.progress("Compression configurations generated")
            stage.add_artifact("compression_configs.json")

        # Mark config generation as completed
        if stage_manager:
            stage_manager.mark_stage_completed("compression_configs_generated", {
                "num_configs": len(compression_configs),
                "fast_mode": fast_mode,
                "max_candidates": max_candidates
            })
    else:
        # Load existing compression configs
        compression_configs_path = output_path / "compression_configs.json"
        with open(compression_configs_path) as f:
            compression_configs = json.load(f)
        print(f"Loaded existing compression configs: {len(compression_configs)} variants")

        # Create empty decision trace for consistency (existing runs)
        decision_trace = DecisionTrace(probe_rank=config.get("lora", {}).get("probe_r", 32))

    # Write compression configs to JSON for debugging/inspection
    compression_configs_path = output_path / "compression_configs.json"
    with open(compression_configs_path, 'w') as f:
        json.dump(compression_configs, f, indent=2, ensure_ascii=False)

    print(f"Compression configs generated:")
    for variant, config_data in compression_configs.items():
        status = config_data["status"]
        if status == "ready":
            actual_r = config_data["actual_r"]
            print(f"  ✅ {variant}: r={actual_r}")
        else:
            reason = config_data.get("reason", "Unknown reason")
            print(f"  ❌ {variant}: {status} - {reason}")

    # Step 3.5: Train and evaluate compressed variants
    print("\nStep 3.5: Training and evaluating compressed variants...")
    variant_results = run_all_compressed_variants(
        config_path=config_path,
        output_dir=output_path,
        compression_configs=compression_configs,
        smoke=smoke,
        stage_manager=stage_manager,
        resume=resume
    )

    # Step 3.6: Compute verdicts
    verdict_analysis = compute_verdicts(
        probe_results=probe_results,
        variant_results=variant_results,
        config=config,
        output_path=output_path,
        smoke=smoke
    )

    # Write verdict analysis to JSON
    verdict_path = output_path / "verdicts.json"
    with open(verdict_path, 'w') as f:
        json.dump(verdict_analysis, f, indent=2, ensure_ascii=False)

    # Step 3.7: Update policy scoreboard with results
    try:
        from gradience.vnext.policy_scoreboard import PolicyScoreboard, create_policy_result_from_bench_data

        scoreboard = PolicyScoreboard()

        # Extract policy results from verdict analysis and compression configs
        config_name = config.get("name", "unknown_config")
        model_name = config.get("model", {}).get("model_name", "unknown_model")
        task_name = config.get("task", {}).get("task_name", "unknown_task")

        policy_results = []

        # Process each compression variant that was tested
        for variant_name, variant_config in compression_configs.items():
            if variant_config.get("status") != "ready":
                continue

            # Get policy information from variant config
            policy_type = variant_config.get("policy_type", "unknown")
            if policy_type == "unknown":
                continue  # Skip non-policy variants

            suggested_rank = variant_config.get("suggested_r", 0)
            actual_rank = variant_config.get("actual_r", 0)

            # Get performance results from verdict analysis
            variant_verdict = verdict_analysis.get("verdicts", {}).get(variant_name)
            if variant_verdict:
                passed = variant_verdict.get("verdict") == "PASS"
                performance_delta = variant_verdict.get("performance_delta", 0.0)

                # Collect all performance results for optimal rank calculation
                all_results = {}
                for vname, vverdict in verdict_analysis.get("verdicts", {}).items():
                    if vverdict.get("performance_delta") is not None:
                        all_results[vname] = vverdict.get("performance_delta", 0.0)

                # Create policy result
                policy_result = create_policy_result_from_bench_data(
                    config_name=config_name,
                    model_name=model_name,
                    task_name=task_name,
                    policy_name=policy_type,
                    suggested_rank=suggested_rank,
                    actual_rank=actual_rank,
                    performance_delta=performance_delta,
                    passed=passed,
                    all_results=all_results,
                    seed=config.get("train", {}).get("seed", 42)
                )

                policy_results.append(policy_result)

        # Add results to scoreboard
        if policy_results:
            scoreboard.add_benchmark_results(config_name, model_name, task_name, policy_results)
            print(f"📊 Updated policy scoreboard with {len(policy_results)} policy results")

            # Export snapshot to output directory
            snapshot_path = output_path / "policy_scoreboard_snapshot.json"
            scoreboard.export_snapshot(snapshot_path)

    except ImportError:
        print("⚠️  Policy scoreboard not available (vnext module not found)")
    except Exception as e:
        print(f"⚠️  Policy scoreboard update failed: {e}")

    # Load audit data for canonical report
    probe_audit_path = output_path / f"probe_r{probe_rank}" / "audit.json"
    with open(probe_audit_path, 'r') as f:
        audit_data = json.load(f)

    # Create canonical bench.json report
    seed = config.get("train", {}).get("seed", 42)
    with monitor_generation("generate_report", output_dir=output_path, seed=seed) as stage:
        stage.progress("Creating canonical benchmark report")
        canonical_report = create_canonical_bench_report(
            probe_results=probe_results,
            variant_results=variant_results,
            verdict_analysis=verdict_analysis,
            audit_data=audit_data,
            compression_configs=compression_configs,
            config=config,
            output_dir=output_path,
            decision_trace=decision_trace
        )

        # Write canonical benchmark report
        stage.progress("Writing JSON report")
        report_path = output_path / "bench.json"
        with open(report_path, 'w') as f:
            json.dump(canonical_report, f, indent=2, ensure_ascii=False)
        stage.add_artifact(report_path)

        # Create and write markdown report
        stage.progress("Generating markdown report")
        markdown_content = create_markdown_report(
            canonical_report=canonical_report,
            config=config,
            output_dir=output_path
        )
        markdown_path = output_path / "bench.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        stage.add_artifact(markdown_path)
        stage.progress("Reports generated successfully")

    # Also write comprehensive internal report for debugging
    internal_report = {
        "bench_version": config.get("bench_version", "0.1"),
        "model": config["model"]["name"],
        "task": f"{config['task']['dataset']}/{config['task']['subset']}",
        "config_path": str(config_path),
        "output_dir": str(output_path),
        "smoke_mode": smoke,
        **probe_results,
        "compression_configs": compression_configs,
        "variants": variant_results,
        "verdicts": verdict_analysis
    }

    internal_report_path = output_path / "bench_internal.json"
    with open(internal_report_path, 'w') as f:
        json.dump(internal_report, f, indent=2, ensure_ascii=False)

    print("\nSteps 3.1-3.6 complete!")
    print("  ✅ Probe trained and telemetry written")
    print("  ✅ Evaluation results written to eval.json")
    print("  ✅ Audit completed and results written to audit.json")
    print("  ✅ Compression configurations generated")
    print("  ✅ Compressed variants trained and evaluated")
    print("  ✅ Verdicts computed and best compression identified")
    print(f"  📊 Canonical report written to: {report_path}")
    print(f"  📝 Human report written to: {markdown_path}")

    # Artifact hygiene cleanup
    run_artifact_hygiene_cleanup(output_path, config)

    print("\nBench protocol complete! 🎉")

    return canonical_report
