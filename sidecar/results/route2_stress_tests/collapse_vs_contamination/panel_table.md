# Collapse vs Contamination Replication Panel

| target_id | source_case_id | expected_channel | replication_type | slice_spec | reason selected |
|---|---|---|---|---|---|
| R1_FR02_case | FR-02 | collapse_like | case | full_500_examples | Nearby collapse lineage case with weaker source-B quality; tests case-level robustness. |
| R2_FR01_even_slice | FR-01 | collapse_like | slice | deterministic_even_indices_mod2 | Deterministic perturbation of collapse anchor; tests slice stability without retraining. |
| R3_CT01_even_slice | CT-01 | contamination_like | slice | deterministic_even_indices_mod2 | Deterministic perturbation of contamination anchor; tests confident-misassignment stability. |
| R4_CT01_odd_slice | CT-01 | contamination_like | slice | deterministic_odd_indices_mod2 | Complementary contamination slice to avoid single-slice overfit. |
