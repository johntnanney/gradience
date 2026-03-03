#!/usr/bin/env python3
"""
Study 13: Figure generation for grokking DFA analysis.

Requires: study13_windowed_alphas.pkl from analyze_study13.py

Figures:
  1. Windowed α timeline for 3 representative seeds
  2. Paired violin/box plots of α_pre vs α_post
  3. Early-warning lead-time histogram
  4. Raw signal overview (one representative seed)
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')

# ── Colors ─────────────────────────────────────────────────────────────
COLORS = {
    'lambda1': '#2196F3',
    'trace_H': '#4CAF50',
    'gHg': '#9C27B0',
    'weight_norm': '#FF9800',
    'grad_norm': '#F44336',
}

NICE = {
    'lambda1': 'λ₁',
    'trace_H': 'Tr(H)',
    'gHg': 'gᵀHg',
    'weight_norm': '‖W‖',
    'grad_norm': '‖∇L‖',
}

CURVATURE = ['lambda1', 'trace_H', 'gHg']
CONTROL = ['weight_norm', 'grad_norm']


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def pick_representative_seeds(seed_results, n=3):
    """Pick n seeds that grokked, sorted by grok_step."""
    grokked = [(sid, sr) for sid, sr in seed_results.items()
                if sr["grok_step"] is not None]
    grokked.sort(key=lambda x: x[1]["grok_step"])

    if len(grokked) <= n:
        return [sid for sid, _ in grokked]

    # Pick first, middle, last
    indices = [0, len(grokked) // 2, len(grokked) - 1]
    return [grokked[i][0] for i in indices]


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Windowed α timeline
# ═══════════════════════════════════════════════════════════════════════

def fig1_windowed_timeline(data, out_dir):
    seed_results = data["seed_results"]
    seeds = pick_representative_seeds(seed_results, n=3)

    if not seeds:
        print("  No grokked seeds for Fig 1")
        return

    fig, axes = plt.subplots(len(seeds), 1, figsize=(14, 4 * len(seeds)),
                              sharex=False)
    if len(seeds) == 1:
        axes = [axes]

    for ax, seed_id in zip(axes, seeds):
        sr = seed_results[seed_id]
        grok_step = sr["grok_step"]

        for metric in CURVATURE + CONTROL:
            segments = sr["windowed"].get(metric, {})
            all_windows = (segments.get("pre", []) +
                          segments.get("transition", []) +
                          segments.get("post", []))
            all_windows.sort(key=lambda w: w["center_step"])

            steps = [w["center_step"] for w in all_windows if w["alpha"] is not None]
            alphas = [w["alpha"] for w in all_windows if w["alpha"] is not None]

            if steps:
                lw = 2.0 if metric in CURVATURE else 1.0
                alpha_vis = 1.0 if metric in CURVATURE else 0.5
                ax.plot(steps, alphas, '-o', color=COLORS[metric],
                        label=NICE[metric], markersize=3, linewidth=lw,
                        alpha=alpha_vis)

        # Grokking line
        if grok_step:
            ax.axvline(x=grok_step, color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'T_grok={grok_step}')

        # White noise reference
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
        ax.text(ax.get_xlim()[0] + 500, 0.52, 'α=0.5 (white noise)',
                fontsize=7, color='gray')

        ax.set_ylabel('DFA α', fontsize=11)
        ax.set_title(f'{seed_id} (grok @ step {grok_step})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', ncol=3)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Training Step', fontsize=11)
    fig.suptitle('Windowed DFA Exponents Across Grokking Transition',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    path = out_dir / "study13_fig1_windowed_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Fig 1 saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Pre vs Post violin/box plots
# ═══════════════════════════════════════════════════════════════════════

def fig2_pre_post_comparison(data, out_dir):
    test_results = data["test_results"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for ax, metric in zip(axes, CURVATURE):
        key = f"test1_{metric}"
        if key not in test_results:
            ax.set_title(f'{NICE[metric]} — no data')
            continue

        tr = test_results[key]
        pre = tr.get("alpha_pre", [])
        post = tr.get("alpha_post", [])

        if not pre or not post:
            ax.set_title(f'{NICE[metric]} — no data')
            continue

        # Box + strip plot
        positions = [0, 1]
        bp = ax.boxplot([pre, post], positions=positions, widths=0.5,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#B3D9FF')
        bp['boxes'][1].set_facecolor('#FFB3B3')

        # Individual points + paired lines
        for p_val, post_val in zip(pre, post):
            ax.plot([0, 1], [p_val, post_val], '-', color='gray',
                    alpha=0.3, linewidth=0.8)

        jitter_pre = np.random.normal(0, 0.05, len(pre))
        jitter_post = np.random.normal(1, 0.05, len(post))
        ax.scatter(jitter_pre, pre, color=COLORS[metric], s=25,
                   zorder=3, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.scatter(jitter_post, post, color=COLORS[metric], s=25,
                   zorder=3, alpha=0.7, edgecolor='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-grok', 'Post-grok'], fontsize=10)

        # Annotation
        p = tr.get("p_one_sided", 1.0)
        d = tr.get("cohens_d", 0.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f'{NICE[metric]}\nΔα={tr["mean_delta"]:+.3f}, d={d:.2f}, p={p:.3g} {sig}',
                     fontsize=10, fontweight='bold')

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.15, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Mean DFA α', fontsize=11)
    fig.suptitle('DFA Exponent Shift: Pre-Grok vs Post-Grok',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    path = out_dir / "study13_fig2_pre_post.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Fig 2 saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Early-warning lead times
# ═══════════════════════════════════════════════════════════════════════

def fig3_early_warning(data, out_dir):
    test_results = data["test_results"]
    ew = test_results.get("test5_early_warning", {})
    lead_times = ew.get("lead_times", [])

    if not lead_times:
        print("  No early-warning data for Fig 3")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for metric in CURVATURE:
        leads = [lt["lead_steps"] for lt in lead_times
                 if lt["metric"] == metric and lt["detect_step"] is not None]
        if leads:
            ax.hist(leads, bins=15, alpha=0.5, color=COLORS[metric],
                    label=f'{NICE[metric]} (n={len(leads)})', edgecolor='white')

    ax.axvline(x=500, color='red', linestyle='--', linewidth=1.5,
               label='500-step threshold')
    ax.set_xlabel('Lead Time (steps before grokking)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Early-Warning Detection Lead Times\n'
                 f'Success rate (≥500 steps): '
                 f'{ew.get("successes_500", 0)}/{ew.get("total_tests", 0)} = '
                 f'{ew.get("success_rate_500", 0):.0%}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = out_dir / "study13_fig3_early_warning.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Fig 3 saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Raw signals for one representative seed
# ═══════════════════════════════════════════════════════════════════════

def fig4_raw_signals(data, out_dir):
    seed_results = data["seed_results"]
    seeds = pick_representative_seeds(seed_results, n=1)
    if not seeds:
        print("  No grokked seed for Fig 4")
        return

    seed_id = seeds[0]
    steps = data["all_data_steps"].get(seed_id, [])
    signals = data["all_data_signals"].get(seed_id, {})
    grok_step = seed_results[seed_id]["grok_step"]

    if not steps:
        return

    steps = np.array(steps)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel A: Curvature
    for metric in CURVATURE:
        vals = signals.get(metric)
        if vals:
            vals = np.array([v if v is not None else np.nan for v in vals])
            axes[0].semilogy(steps, np.abs(vals) + 1e-15, color=COLORS[metric],
                            label=NICE[metric], linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel('Curvature (|value|, log)', fontsize=10)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title(f'Raw Telemetry: {seed_id} (grok @ step {grok_step})',
                     fontsize=12, fontweight='bold')

    # Panel B: Norms
    for metric in CONTROL:
        vals = signals.get(metric)
        if vals:
            vals = np.array([v if v is not None else np.nan for v in vals])
            axes[1].plot(steps, vals, color=COLORS[metric],
                        label=NICE[metric], linewidth=0.7, alpha=0.8)
    axes[1].set_ylabel('Norms', fontsize=10)
    axes[1].legend(fontsize=8, loc='upper right')

    # Panel C: Val accuracy (to show grokking transition)
    # Need to get from all records, not just Hessian records
    # For simplicity, use the signals available
    # We can add val_acc if we pass all_records through
    axes[2].text(0.5, 0.5, '(val_acc requires all_records — see telemetry.jsonl)',
                transform=axes[2].transAxes, ha='center', fontsize=10, color='gray')
    axes[2].set_ylabel('Val Accuracy', fontsize=10)
    axes[2].set_xlabel('Training Step', fontsize=11)

    for ax in axes:
        if grok_step:
            ax.axvline(x=grok_step, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = out_dir / "study13_fig4_raw_signals.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Fig 4 saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="results/study13/study13_windowed_alphas.pkl")
    parser.add_argument("--out_dir", type=str, default="results/study13/figures")
    args = parser.parse_args()

    data = load_data(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Study 13 figures...")
    fig1_windowed_timeline(data, out_dir)
    fig2_pre_post_comparison(data, out_dir)
    fig3_early_warning(data, out_dir)
    fig4_raw_signals(data, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
