# N134 Spec v3.1 Update — GPU Platform Amendment (Archived)

**This document is archived.** It contains the edit instructions that were applied to `sidecar/notes/n134_spec.md` to produce spec v3.1 from spec v3.

**Status:** Applied April 19 2026. The active spec is `sidecar/notes/n134_spec.md` at v3.1.

**Scope of changes applied:**

1. Status line updated v3 → v3.1.
2. Version history extended with v3.1 entry.
3. §9.4 resource estimate rewritten. GPU target changed from single H100 (~18 h, ~$44) to single RTX 6000 Ada 48GB on RunPod Secure Cloud (~42 h, ~$31).
4. §9.4 extended with VRAM footprint breakdown (~20–22 GB peak), cost comparison across GPUs on the efficient frontier, acceptable substitutes (L40S, RTX 6000 Pro, A100 80GB, H100 PCIe) in priority order, unacceptable configurations flagged as material deviations, and phased execution recommendation.
5. Appendix B (Change log) extended with v3→v3.1 subsection.
6. Closing line updated to reflect v3.1 supersedes v3.

**What was NOT changed (verified):**

- §2 (confounds C1–C4): unchanged.
- §3 (experimental design — model, task set, seeds, training protocol, audit schema): unchanged. bf16 precision works identically on both H100 and RTX 6000 Ada.
- §4 (H1 score, decision rule): unchanged.
- §5 (statistical protocol): unchanged.
- §6 (four-method scheduled comparison): unchanged.
- §7 (deviation policy): unchanged.
- §8 (outcome interpretations): unchanged.
- §9.1 (directory layout): unchanged.
- §9.2 (execution order / coding-agent checklist): unchanged.
- §9.3 (external dependencies): unchanged. All packages (PEFT, Gradience, KnOTS, TSV, SVC) work identically on 6000 Ada.
- §9.5 (required artifacts in final report): unchanged.
- Appendix A (known unknowns): unchanged. Four bullets preserved.
- `scripts/n134/` (all 9 scripts): unchanged.
- `sidecar/data/n134/audit_v2_1.schema.json`: unchanged.

**Rationale:** LoRA training on a 7B frozen base model is memory-bandwidth-bound, not FLOPS-bound; 48 GB VRAM is sufficient for Mistral-7B + rank-16 LoRA in bf16 with headroom to spare; RTX 6000 Ada on Secure Cloud at $0.74/hr delivers the entire protocol for ~$31 against ~$44 for the H100 PCIe path and ~$42 for A100 80GB.

**Operational note for launching RunPod Secure Cloud pods:**

- Template: any PyTorch 2.x template with CUDA 12.x. "PyTorch 2.4.0" is a tested baseline.
- Container disk: 50 GB minimum.
- Volume: not required; ~2 GB of persistent output can be downloaded before teardown.
- Region: any region with 6000 Ada availability.
- vCPUs / RAM: default pairing (~10 vCPUs, 167 GB RAM) is more than sufficient. Audit phase benefits from multiple cores for parallel SVD; do not undersize.
- RunPod API GPU type identifier: `NVIDIA RTX 6000 Ada Generation`.

---

*Archive retained for audit trail. See `sidecar/notes/n134_spec.md` for the active pre-registration.*
