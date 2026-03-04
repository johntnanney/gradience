#!/usr/bin/env python3
"""
Prototype the TrainingMonitor against Study 7 telemetry.

Study 7: Modular addition mod 97, 200K steps, memorised but never grokked (wd=0.001).
Training loss hit a floor early. Monitor should detect this and recommend stopping.

Usage:
    python scripts/prototype_monitor.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience.vnext.monitor import MonitorConfig
from gradience.vnext.monitor_replay import replay_telemetry


def main():
    # Study 7 telemetry
    telemetry_path = Path(__file__).resolve().parent.parent.parent / (
        "Gradience II/archive/signals_telemetry_validation_v1/"
        "results/grok_modadd_study7a_gpu/telemetry.jsonl"
    )

    if not telemetry_path.exists():
        print(f"Telemetry file not found: {telemetry_path}")
        print("Trying alternative path...")
        # Try relative to mnt
        alt = Path("/sessions/trusting-magical-allen/mnt/Gradience II/archive/"
                    "signals_telemetry_validation_v1/results/"
                    "grok_modadd_study7a_gpu/telemetry.jsonl")
        if alt.exists():
            telemetry_path = alt
        else:
            print("Could not find Study 7 telemetry. Exiting.")
            sys.exit(1)

    print(f"Replaying: {telemetry_path}")
    print(f"File size: {telemetry_path.stat().st_size / 1024:.1f} KB")
    print()

    # Default config
    config = MonitorConfig(
        window_size=50,
        loss_plateau_patience=5,
        loss_plateau_delta=0.01,
        grad_plateau_patience=5,
        grad_plateau_delta=0.02,
        grad_spike_multiplier=5.0,
        eval_patience=5,
        alert_cooldown_steps=100,  # 100 telemetry steps between alerts
    )

    result = replay_telemetry(telemetry_path, config)

    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    print()

    # Show all alerts with details
    if result.alerts:
        print("Detailed alerts:")
        print("-" * 60)
        for a in result.alerts:
            print(f"  Step {a.step:>7d}  [{a.severity:>7s}]  {a.code}")
            print(f"             {a.message}")
            if a.recommendation:
                print(f"             -> {a.recommendation[:80]}...")
            print()
    else:
        print("No alerts fired. Consider relaxing thresholds.")

    # Summary statistics
    print("Monitor statistics:")
    print(f"  Training steps processed: {result.n_steps_processed}")
    print(f"  Evaluations processed:    {result.n_evals_processed}")
    print(f"  Step range:               {result.first_step} -> {result.last_step}")

    if result.would_have_stopped:
        total_steps = (result.last_step or 0) - (result.first_step or 0)
        saved = (result.last_step or 0) - result.would_have_stopped_step
        print(f"\n  WOULD HAVE STOPPED at step {result.would_have_stopped_step}")
        if total_steps > 0:
            print(f"  Training saved: {saved / total_steps * 100:.0f}% ({saved} steps)")


if __name__ == "__main__":
    main()
