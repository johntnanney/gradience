# Lock Notes — Benchmark Reliability Study v1.1-LOCKED

**Lock date:** 2026-04-25
**Pre-registration:** `preregistration/prereg_v1_1_LOCKED.md`
**Config hash:** `65fdd1c2` (full SHA-256: `65fdd1c29d63a78b85c128d56888240967c1b7cf5274cedf8bfc01cc7a0242d9`)
**Validation status:** PASS (0 errors, 0 warnings)
**Validation report:** `reports/config_validation.json`

This document is the lock-audit record. It enumerates every spec-committed parameter, its value at lock time, and its provenance. Subsequent changes to any of these parameters require a `deviations.md` entry and are not the work of this lock.

---

## 1. Models pinned

| model_id | hf_name | hf_revision | role |
|---|---|---|---|
| pythia_410m | EleutherAI/pythia-410m | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` | primary, base |
| pythia_1_4b | EleutherAI/pythia-1.4b | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | primary, base |
| qwen2_5_1_5b_instruct | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | primary, instruction-tuned |
| mistral_7b_v0_3 | mistralai/Mistral-7B-v0.3 | TODO_LOCK_AT_EXTENSION_TIME | optional 7B extension; budget-contingent; not part of v1.1 lock |

The optional 7B extension is intentionally not pinned at v1.1 lock because it is not part of the primary-hypothesis tests and its execution depends on budget availability. If the extension is later run, its `hf_revision` must be pinned at that time and the run logged as a deviation-supplement.

## 2. Datasets pinned

| benchmark_id | hf_dataset | hf_config | dataset_version_hash |
|---|---|---|---|
| arc_challenge | allenai/ai2_arc | ARC-Challenge | `210d026faf9955653af8916fad021475a3f00453` |
| hellaswag | Rowan/hellaswag | (default) | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| truthfulqa_mc | truthful_qa | multiple_choice | `741b8276f2d1982aa3d5b832d3ee81ed3b896490` |
| mmlu_panel | cais/mmlu | (default) | `c30699e8356da336a370243923dbaf21066bb9fe` |
| winogrande | winogrande | winogrande_xl | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` |
| gsm8k | gsm8k | main | `740312add88f781978c0658806c59bc2815b9866` |

## 3. MMLU subject panel pinned

The five subjects are locked at v1.1. Substitutions made from the v1 proposal during v1.1-draft (2026-04-24, see preregistration §14.1):

- ✓ world_religions (humanities) — unchanged from v1
- elementary_mathematics (STEM) — substituted for high_school_mathematics
- high_school_psychology (social sciences) — specified from generic "psychology"
- professional_accounting (professional) — substituted for professional_medicine
- ✓ global_facts (cross-domain) — unchanged from v1

Substitution rationale: small-model floor risk on STEM and professional slots. See preregistration §5.2.

## 4. Prompts pinned

24 prompt files at `prompts/<benchmark_id>/<prompt_id>.txt`. All entries in `configs/prompts.yaml` have `admissibility_status: locked` and content hashes recorded.

**Provenance summary:**
- P1_original: benchmark-author sources (paper + GitHub repo cite)
- P2_lm_eval: `EleutherAI/lm-evaluation-harness` at commit `c1c4bea3777f73e188395264083adcf454913344` (main HEAD on 2026-04-08)
- P3_helm_or_published: `stanford-crfm/helm` at commit `11937097bd9534e538eaaa31b21197086fc1a113` (main HEAD on 2026-04-23) for 5 of 6 benchmarks; Winogrande P3 sources from Brown et al. 2020 GPT-3 paper App. G.7 because HELM has no English Winogrande scenario
- P4_minimal_sourced: author-constructed minimal variants; declared in `preregistration/appendices/admissibility_sources_LOCKED.md`

**Byte-identical-template identity classes** (documented in `configs/prompts.yaml` `notes` fields):

| identity class | size | members |
|---|---|---|
| 1 (P4 minimal) | 5 | arc_challenge, hellaswag, mmlu_panel, truthfulqa_mc, winogrande P4_minimal_sourced |
| 2 (HELM MCQ adapter) | 3 | arc_challenge, hellaswag, truthfulqa_mc P3_helm_or_published |
| 3 (harness Q/Choices/A) | 3 | hellaswag P1_original, hellaswag P2_lm_eval, mmlu_panel P2_lm_eval |
| 4 (ARC author-harness) | 2 | arc_challenge P1_original, arc_challenge P2_lm_eval |
| 5 (Winogrande author-harness) | 2 | winogrande P1_original, winogrande P2_lm_eval |
| 6 (GSM8K author-harness) | 2 | gsm8k P1_original, gsm8k P2_lm_eval |

The byte-collapse is itself a finding about the canonical-source ecology and is reported as such in the manuscript. The variance-components decomposition uses the nominal P-id labels; effective DOF is reduced for benchmarks whose P1/P2 collapse (4 of 6).

## 5. Decision rules and thresholds pinned

From `configs/analysis_config.yaml`:

| parameter | value | source |
|---|---|---|
| H1 tolerance threshold | 0.005 (single-occasion CI lower bound) | preregistration §3.5, §8.8 |
| H1 benchmarks required | 3 of 5 | preregistration §3.2 |
| H2 generalizability threshold | 0.80 | preregistration §3.3 |
| H3 ranking-reversal threshold | 0.20 | preregistration §3.3, §14.5 (raised from 0.10 at v1.1-draft) |
| H4 MMLU interaction threshold | 0.10 | preregistration §3.3 |
| Bootstrap n_resamples | 10000 | preregistration §9.3, SPEC §4.3 |
| Bootstrap random_seed | 20260424 | seed-pinned for reproducibility |
| Mixed-effects cascade | 4 levels | preregistration §10.1, SPEC §10.1 |
| Convergence: gradient_norm_above | 1.0e-3 | SPEC §10.1 |

## 6. Few-shot seeds pinned

- Seeds: 42, 123, 2024
- Drawing protocol: `numpy.random.default_rng(seed).choice(n, size=k, replace=False)` from each benchmark's `fewshot_source_split`
- 0-shot benchmark (TruthfulQA-MC): seed_facet collapsed to single level, `seed_id="0shot"`
- Per-subject draws for MMLU panel
- Hard-fail leakage check against eval-split item IDs (preregistration §3.7, SPEC §8.3)

## 7. Total nominal condition count

Per `00_validate_config.py`: **600 nominal conditions** in the primary tier. Decomposition:

- arc_challenge: 3 models × 1 subject × 4 prompts × 3 seeds × 2 scoring rules = 72
- hellaswag: 72
- truthfulqa_mc: 3 × 1 × 4 × 1 × 2 = 24 (0-shot, single-seed)
- mmlu_panel: 3 × 5 × 4 × 3 × 2 = 360
- winogrande: 72
- **Primary total: 600**

Secondary tier (GSM8K): 3 models × 4 prompts × 3 seeds × 2 extraction variants = 72.

Per-cell condition counts (subject of variance-components decomposition): 24 nominal for benchmarks with seed_facet=true, 8 for TruthfulQA-MC (0-shot collapses seeds).

## 8. Software environment

| dependency | version constraint | locked at v1.1 |
|---|---|---|
| Python | >=3.11 | 3.11.x recommended |
| pyyaml | >=6.0,<7 | per pyproject.toml |
| pandas | >=2.1,<3 | per pyproject.toml |
| pyarrow | >=14,<17 | per pyproject.toml |
| numpy | >=1.26,<2 | per pyproject.toml |
| scipy | >=1.11,<2 | per pyproject.toml |
| statsmodels | >=0.14,<0.16 | per pyproject.toml |
| jsonschema | >=4.20,<5 | per pyproject.toml |
| datasets (HF) | >=2.14,<3 | per pyproject.toml |
| matplotlib | >=3.8,<4 | per pyproject.toml |
| seaborn | >=0.13,<0.14 | per pyproject.toml |

Exact version pins to be recorded in a `requirements.lock` file at execution time.

## 9. Ancillary files at lock

- `SPEC_CPU_v0_2.md` — CPU-side spec; defines pipeline contract
- `IMPLEMENTATION_DEVIATIONS.md` — 13 deviations from spec documented
- `preregistration/appendices/admissibility_sources_LOCKED.md` — P4 minimal-prompt justifications and source-URL pinning
- 13 pipeline scripts (`scripts/00`–`10`, `scripts/98_reproducibility_trace.py`, `scripts/99_make_report.py`) — implemented and tested, 133/133 tests passing

## 10. Lock procedure executed

1. ✓ Production configs created at `configs/*.yaml` with all real values
2. ✓ Prompt content hashes computed via SHA-256 on each prompt file; recorded in `configs/prompts.yaml`
3. ✓ All 24 prompt entries promoted from `admissibility_status: draft` to `admissibility_status: locked`
4. ✓ `00_validate_config.py` run: PASS, 0 errors, 0 warnings
5. ✓ Preregistration upgraded from `preregistration_v1.md` (v1.1-draft) to `preregistration/prereg_v1_1_LOCKED.md` (v1.1-LOCKED)
6. ✓ This `LOCK_NOTES.md` audit record created

**Remaining (user-side, post-lock):**

7. Run pytest to verify the 133-test suite still passes against the locked configs
8. `git add` all locked artifacts; `git commit -m "papers/benchmark_reliability_study: lock v1.1 pre-registration"`
9. `git tag v1_1_LOCKED -m "Pre-registration locked; subsequent changes require deviations.md entry"`
10. (Optional) `git push --tags`

## 11. What this lock commits to

- The five primary benchmarks listed in §2.
- The MMLU subject panel listed in §3.
- The 24 prompt files listed in §4.
- The decision rules and thresholds listed in §5.
- The few-shot seeds and protocol listed in §6.
- The 600 primary + 72 secondary nominal conditions.
- The 9 hash-pinned external dependencies (3 models + 6 datasets).

## 12. What this lock does not commit to

- Optional 7B extension (`mistral_7b_v0_3`) — may be pinned at execution time if budget allows
- GPU-side inference backend implementation — out of CPU-spec scope
- Exact Python package versions beyond the constraints in `pyproject.toml` — to be locked in `requirements.lock` at execution time
- Manuscript writing — happens after analysis

## 13. Reproducibility commitment

Any party with this repository at tag `v1_1_LOCKED`, the locked HF model and dataset revisions, and a Python environment satisfying the dependency constraints can:

- Build the condition manifest deterministically: `01_build_manifests.py` produces byte-identical CSV.
- Verify prompt content via `03_validate_prompts.py`: each file's SHA-256 must match `configs/prompts.yaml`.
- Execute the pipeline end-to-end after GPU-side inference produces the raw runs.
- Reproduce every headline number bit-identically (modulo bootstrap stochasticity, which is seed-pinned).

Reviewer audit path: tag → repository state → config_hash `65fdd1c2` → headline numbers in manuscript Table 4.

---

*End of lock notes. v1.1-LOCKED at 2026-04-25.*
