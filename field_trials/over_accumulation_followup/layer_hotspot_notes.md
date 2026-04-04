# Layer Hotspot Notes

### Case 1 — High-overlap low-score reference

- Pair: `p3_retained_sst2`
- Cohort: `group_1_high_overlap_low_proxy`
- Outcome: merged=0.8200, delta_vs_best=-0.0660
- Structure: mean_overlap=0.2537, max_oa_score=0.1396, advisory=none
- Top layer hotspots:
  - `base_model.model.distilbert.transformer.layer.2.attention.q_lin` score=0.1396 band=low (alignment=0.292, concentration=0.550, coefficient=0.307)
  - `base_model.model.distilbert.transformer.layer.1.attention.q_lin` score=0.1285 band=low (alignment=0.281, concentration=0.531, coefficient=0.288)
  - `base_model.model.distilbert.transformer.layer.0.attention.v_lin` score=0.1254 band=low (alignment=0.338, concentration=0.432, coefficient=0.231)

### Case 2 — High-overlap high-score proxy-elevated

- Pair: `ca_family_sst2xIMDB_eval_imdb`
- Cohort: `group_2_high_overlap_elevated_proxy`
- Outcome: merged=0.8680, delta_vs_best=-0.0160
- Structure: mean_overlap=0.3085, max_oa_score=0.3534, advisory=none
- Top layer hotspots:
  - `base_model.model.distilbert.transformer.layer.5.attention.v_lin` score=0.3534 band=watch (alignment=0.558, concentration=0.476, coefficient=1.000)
  - `base_model.model.distilbert.transformer.layer.4.attention.v_lin` score=0.2732 band=low (alignment=0.398, concentration=0.551, coefficient=1.000)
  - `base_model.model.distilbert.transformer.layer.3.attention.v_lin` score=0.2321 band=low (alignment=0.341, concentration=0.545, coefficient=1.000)

### Case 3 — Non-overlap pathology anchor

- Pair: `p3_control_sst2_x_agnews`
- Cohort: `group_3_non_overlap_pathology_anchor`
- Outcome: merged=0.8380, delta_vs_best=-0.0740
- Structure: mean_overlap=0.1130, max_oa_score=0.0444, advisory=none
- Top layer hotspots:
  - `base_model.model.distilbert.transformer.layer.5.attention.v_lin` score=0.0444 band=low (alignment=0.099, concentration=0.210, coefficient=1.000)
  - `base_model.model.distilbert.transformer.layer.0.attention.q_lin` score=0.0432 band=low (alignment=0.111, concentration=0.364, coefficient=0.454)
  - `base_model.model.distilbert.transformer.layer.3.attention.v_lin` score=0.0360 band=low (alignment=0.072, concentration=0.281, coefficient=1.000)

## Interpretation
- In this first pass, hotspot ranking is interpretable, but official pair-level advisory may remain non-triggered.
- Hotspot notes are diagnostic context, not causal proof.
