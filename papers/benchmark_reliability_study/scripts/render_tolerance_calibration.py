"""
Regenerate tolerance_calibration.pdf using rule-conditioned tolerances.

X-axis: rule-conditioned single-occasion tolerance (k=1)
Y-axis: rule-conditioned full-design tolerance (averaged over rule-conditioned
        admissible conditions: 12 on ARC/HellaSwag/Winogrande, 60 on MMLU,
        4 on TruthfulQA-MC)
Both axes log-scaled. Cells colored by regime.
"""
import csv, math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vc = defaultdict(dict)
with open('analysis/variance_components/aggregate_vc.csv') as f:
    for r in csv.DictReader(f):
        vc[(r['benchmark_id'], r['model_id'])][r['source']] = float(r['variance'])

conds_by_cell = defaultdict(list)
with open('runs/normalized/condition_level_primary.csv') as f:
    for r in csv.DictReader(f):
        bmr = (r['benchmark_id'], r['model_id'], r['scoring_rule_id'])
        try:
            acc = float(r['accuracy'])
        except (ValueError, TypeError):
            continue
        try:
            par = float(r['parseability_rate']) if r['parseability_rate'] else None
        except ValueError:
            par = None
        conds_by_cell[bmr].append((acc, par))

THRESHOLD = 0.30

def compute_cell(b, m, s):
    parseabilities = [p for (_, p) in conds_by_cell[(b,m,s)] if p is not None]
    if not parseabilities:
        regime = 'g'
    else:
        from statistics import median
        regime = 'pf' if median(parseabilities) < THRESHOLD else 'g'
    n_w = len(conds_by_cell[(b,m,s)])
    if regime == 'pf':
        accs = [a for (a, _) in conds_by_cell[(b,m,s)]]
        mean_acc = sum(accs)/len(accs)
        sd = (sum((x - mean_acc)**2 for x in accs) / (len(accs)-1)) ** 0.5
        sem_single = sd
    else:
        vp = vc[(b,m)].get('prompt',0.0); vs = vc[(b,m)].get('seed',0.0); ve = vc[(b,m)].get('residual',0.0)
        sem_single = math.sqrt(max(vp+vs+ve, 0.0))
    sem_full = sem_single / math.sqrt(n_w)
    return regime, 2*sem_single, 2*sem_full

cells = []
for b in ['arc_challenge','hellaswag','mmlu_panel','truthfulqa_mc','winogrande']:
    for m in ['pythia_1_4b','pythia_410m','qwen2_5_1_5b_instruct']:
        for s in ['generate_parse','ll_norm']:
            regime, single, full = compute_cell(b, m, s)
            cells.append((b, m, s, regime, single, full))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.set_xscale('log')
ax.set_yscale('log')

g_xs = [c[4] for c in cells if c[3] == 'g']
g_ys = [c[5] for c in cells if c[3] == 'g']
pf_xs = [c[4] for c in cells if c[3] == 'pf']
pf_ys = [c[5] for c in cells if c[3] == 'pf']

ax.scatter(g_xs, g_ys, marker='o', s=60, facecolors='none', edgecolors='black',
           linewidths=1.2, label=r'$g$-theory regime', zorder=3)
ax.scatter(pf_xs, pf_ys, marker='s', s=60, facecolors='black', edgecolors='black',
           linewidths=1.2, label='parse-failure-dominated regime', zorder=3)

ax.axvline(0.005, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(0.005, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axvline(0.05, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axhline(0.05, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

xs = [c[4] for c in cells]; ys = [c[5] for c in cells]
all_vals = xs + ys
lo = min(all_vals) * 0.5; hi = max(all_vals) * 2
diag = [lo, hi]
ax.plot(diag, diag, color='gray', linestyle='-', linewidth=0.7, alpha=0.5,
        label=r'$y = x$ (no averaging gain)', zorder=1)

ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r'Rule-conditioned single-occasion tolerance ($k = 1$)')
ax.set_ylabel('Rule-conditioned full-design tolerance')
ax.set_title('')

ax.text(0.0048, hi/1.4, r'$\pm 0.005$', rotation=90, fontsize=8, va='top', ha='right', alpha=0.7)
ax.text(0.048, hi/1.4, r'$\pm 0.05$', rotation=90, fontsize=8, va='top', ha='right', alpha=0.5)
ax.text(hi/1.4, 0.0048, r'$\pm 0.005$', fontsize=8, va='top', ha='right', alpha=0.7)
ax.text(hi/1.4, 0.048, r'$\pm 0.05$', fontsize=8, va='bottom', ha='right', alpha=0.5)

ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.savefig('figures/tolerance_calibration.pdf', bbox_inches='tight')
print(f"Figure regenerated. {len([c for c in cells if c[5] < 0.005])} cells license two-decimal under full-design averaging.")
print(f"  g-theory cells: {len(g_xs)}, pf cells: {len(pf_xs)}")
print(f"  X range: [{min(xs):.4f}, {max(xs):.4f}]")
print(f"  Y range: [{min(ys):.4f}, {max(ys):.4f}]")
