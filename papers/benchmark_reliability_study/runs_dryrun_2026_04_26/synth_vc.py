"""Synthesize a minimal aggregate_vc.csv for dry-run pipeline exercise.

The dry-run goal is pipeline survivability, not numerical correctness. Script 06's
MixedLM fit on real-scale data (524k items, crossed random effects on item_id with
1172+ levels) exceeded a 15-minute wall-clock budget without producing per-benchmark
progress logs. To exercise scripts 07->10/98/99 plumbing, this script writes a
placeholder aggregate_vc.csv with the schema that 06 emits (per
06_variance_components.py:emit_aggregate_vc, lines 415-451): one row per
(benchmark, model, source) triple, with placeholder variance/proportion values.

THE NUMBERS ARE NOT FOR INTERPRETATION. They exist only to satisfy 07's input
contract.
"""

import sys
import pandas as pd

cond_path = sys.argv[1]
out_path = sys.argv[2]

df = pd.read_csv(cond_path)
cells = df.groupby(["benchmark_id", "model_id"]).size().reset_index(name="n")

sources_template = [
    ("prompt",       0.0010, 0.10),
    ("seed",         0.0005, 0.05),
    ("scoring_rule", 0.0020, 0.20),
    ("residual",     0.0065, 0.65),
]

rows = []
for _, r in cells.iterrows():
    n_cond = int(r["n"])
    df_used = max(n_cond - 1, 0)
    for src, var, prop in sources_template:
        rows.append({
            "benchmark_id": r["benchmark_id"],
            "model_id":     r["model_id"],
            "source":       src,
            "variance":     var,
            "proportion":   prop,
            "df":           df_used,
        })

out = pd.DataFrame(
    rows,
    columns=["benchmark_id", "model_id", "source",
             "variance", "proportion", "df"],
)
out.to_csv(out_path, index=False)
print(f"wrote {len(out)} rows for {len(cells)} (benchmark, model) cells "
      f"to {out_path} (DRYRUN PLACEHOLDER, NOT FOR INTERPRETATION)")
