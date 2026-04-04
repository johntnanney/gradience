#!/usr/bin/env bash
# ============================================================================
# consolidate_research.sh — Consolidate scattered Gradience research materials
#                           into the Gradience II repository
#
# Created:  April 4, 2026
# Purpose:  Copies (not moves) research documents from ~/Downloads,
#           ~/Documents/AI Research, and ~/Desktop into the Gradience II
#           directory structure. Originals are left in place so you can
#           verify everything landed correctly before deleting them.
#
# Usage:    bash scripts/consolidate_research.sh [--dry-run]
#
#           --dry-run   Show what would be copied without doing anything.
#                       Recommended for the first run.
#
# After running:
#   1. Review the output to confirm all copies succeeded
#   2. Spot-check a few files in their new locations
#   3. When satisfied, delete the originals from ~/Downloads etc.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRADIENCE_II="$HOME/Gradience II"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "═══════════════════════════════════════════════════════"
    echo "  DRY RUN — no files will be copied or created"
    echo "═══════════════════════════════════════════════════════"
    echo
fi

# Counters
COPIED=0
SKIPPED=0
ERRORS=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { echo "  [INFO]  $*"; }
log_copy()  { echo "  [COPY]  $*"; }
log_skip()  { echo "  [SKIP]  $*"; }
log_warn()  { echo "  [WARN]  $*"; }
log_error() { echo "  [ERROR] $*"; }
log_mkdir() { echo "  [MKDIR] $*"; }

# safe_copy SOURCE DEST [RENAME]
#   Copies SOURCE to DEST directory (or DEST/RENAME if a third arg is given).
#   Skips if the destination already exists.
#   In dry-run mode, just prints what would happen.
safe_copy() {
    local src="$1"
    local dest_dir="$2"
    local rename="${3:-}"

    if [[ ! -f "$src" ]]; then
        log_warn "Source not found, skipping: $src"
        ((SKIPPED++)) || true
        return
    fi

    local filename
    if [[ -n "$rename" ]]; then
        filename="$rename"
    else
        filename="$(basename "$src")"
    fi

    local dest="$dest_dir/$filename"

    if [[ -f "$dest" ]]; then
        # Check if they're identical
        if diff -q "$src" "$dest" > /dev/null 2>&1; then
            log_skip "Already exists (identical): $dest"
        else
            log_warn "Already exists (DIFFERENT): $dest"
            log_warn "  Source: $src"
            log_warn "  You may want to diff these manually."
        fi
        ((SKIPPED++)) || true
        return
    fi

    if $DRY_RUN; then
        log_copy "$src → $dest"
    else
        cp "$src" "$dest"
        log_copy "$src → $dest"
    fi
    ((COPIED++)) || true
}

# safe_mkdir DIR
#   Creates directory if it doesn't exist.
safe_mkdir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        if $DRY_RUN; then
            log_mkdir "(would create) $dir"
        else
            mkdir -p "$dir"
            log_mkdir "$dir"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "═══════════════════════════════════════════════════════"
echo "  Gradience Research Consolidation"
echo "═══════════════════════════════════════════════════════"
echo

if [[ ! -d "$GRADIENCE_II" ]]; then
    log_error "Gradience II directory not found at: $GRADIENCE_II"
    log_error "Please verify the path and try again."
    exit 1
fi
log_info "Target: $GRADIENCE_II"
echo

# ---------------------------------------------------------------------------
# Create new directories as needed
# ---------------------------------------------------------------------------

echo "── Creating directories ──────────────────────────────"

safe_mkdir "$GRADIENCE_II/docs/specs"
safe_mkdir "$GRADIENCE_II/results/study16_merge_ablation"
safe_mkdir "$GRADIENCE_II/results/study17_compression"
safe_mkdir "$GRADIENCE_II/archive"
safe_mkdir "$GRADIENCE_II/archive/abstracts"
safe_mkdir "$GRADIENCE_II/archive/protocols"
safe_mkdir "$GRADIENCE_II/archive/findings"
safe_mkdir "$GRADIENCE_II/archive/experiment_data"
safe_mkdir "$GRADIENCE_II/archive/philosophy"

echo

# ---------------------------------------------------------------------------
# Tier 1 — Critical documents from ~/Downloads
# ---------------------------------------------------------------------------

echo "── Tier 1: Critical documents (~/Downloads) ─────────"

# Paper draft → docs/papers/
safe_copy \
    "$HOME/Downloads/GRADIENCE_PAPER_DRAFT_REVISED.md" \
    "$GRADIENCE_II/docs/papers"

# Post 7 — check if different from existing version
# Existing: blog_series/SERIES_POST_7_GRADIENCE_011.md
# Downloads: SERIES_POST_7_FINAL.md (may be a later/different edit)
safe_copy \
    "$HOME/Downloads/SERIES_POST_7_FINAL.md" \
    "$GRADIENCE_II/docs/blog_series"

# Post 8 → blog_series/
safe_copy \
    "$HOME/Downloads/POST8_FINAL.md" \
    "$GRADIENCE_II/docs/blog_series" \
    "SERIES_POST_8_FINAL.md"

# Study 16B → results/study16_merge_ablation/
safe_copy \
    "$HOME/Downloads/STUDY16B_DRAFT.md" \
    "$GRADIENCE_II/results/study16_merge_ablation"

# Study 17A → results/study17_compression/
safe_copy \
    "$HOME/Downloads/STUDY17A_INTERIM_RESULTS.md" \
    "$GRADIENCE_II/results/study17_compression"

# DFA workshop paper — already in docs/project/, copy to docs/papers/ too
# for discoverability alongside other papers
safe_copy \
    "$HOME/Downloads/DFA_WORKSHOP_PAPER.md" \
    "$GRADIENCE_II/docs/papers"

# v0.12.0 spec → docs/specs/
safe_copy \
    "$HOME/Downloads/spec-phase-a.md" \
    "$GRADIENCE_II/docs/specs"

# Decision infrastructure PDF → docs/papers/
safe_copy \
    "$HOME/Desktop/Gradience_LoRA_Decision_Infrastructure.pdf" \
    "$GRADIENCE_II/docs/papers"

echo

# ---------------------------------------------------------------------------
# Tier 2 — Important supplementary documents
# ---------------------------------------------------------------------------

echo "── Tier 2: Supplementary documents ───────────────────"

# Executive summary → docs/blog_drafts/
safe_copy \
    "$HOME/Downloads/GRADIENCE_EXECUTIVE_SUMMARY.md" \
    "$GRADIENCE_II/docs/blog_drafts"

# LoRA findings → archive/findings/
safe_copy \
    "$HOME/Downloads/LORA_FINDINGS.md" \
    "$GRADIENCE_II/archive/findings"

# Program abstracts → archive/abstracts/
safe_copy \
    "$HOME/Documents/AI Research/ABSTRACTS.md" \
    "$GRADIENCE_II/archive/abstracts"

echo

# ---------------------------------------------------------------------------
# Tier 2 — Findings and protocols from ~/Documents/AI Research
# ---------------------------------------------------------------------------

echo "── Tier 2: AI Research findings & protocols ──────────"

# Copy all files from Findings_and_Notes/
if [[ -d "$HOME/Documents/AI Research/Findings_and_Notes" ]]; then
    for f in "$HOME/Documents/AI Research/Findings_and_Notes"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/findings"
    done
else
    log_warn "Findings_and_Notes directory not found"
fi

# Copy all files from Protocol_and_Methods/
if [[ -d "$HOME/Documents/AI Research/Protocol_and_Methods" ]]; then
    for f in "$HOME/Documents/AI Research/Protocol_and_Methods"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/protocols"
    done
else
    log_warn "Protocol_and_Methods directory not found"
fi

echo

# ---------------------------------------------------------------------------
# Tier 3 — Archival materials (philosophy, historical code, experiments)
# ---------------------------------------------------------------------------

echo "── Tier 3: Archival materials ────────────────────────"

# Deleuzean AI manuscripts
if [[ -d "$HOME/Documents/AI Research/Manuscripts/Deleuzean_AI" ]]; then
    for f in "$HOME/Documents/AI Research/Manuscripts/Deleuzean_AI"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/philosophy"
    done
else
    log_warn "Deleuzean_AI manuscripts directory not found"
fi

# Philosophy and AI manuscripts
if [[ -d "$HOME/Documents/AI Research/Manuscripts/Philosophy_and_AI" ]]; then
    for f in "$HOME/Documents/AI Research/Manuscripts/Philosophy_and_AI"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/philosophy"
    done
else
    log_warn "Philosophy_and_AI manuscripts directory not found"
fi

# Deleuzean AI analysis scripts
if [[ -d "$HOME/Documents/AI Research/Code/Deleuzean_AI" ]]; then
    safe_mkdir "$GRADIENCE_II/archive/historical_code"
    for f in "$HOME/Documents/AI Research/Code/Deleuzean_AI"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/historical_code"
    done
else
    log_warn "Deleuzean_AI code directory not found"
fi

# Experiment data archives
if [[ -d "$HOME/Documents/AI Research/Experiment_Data/Gradience_Experiments" ]]; then
    for f in "$HOME/Documents/AI Research/Experiment_Data/Gradience_Experiments"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/experiment_data"
    done
else
    log_warn "Gradience_Experiments data directory not found"
fi

# Philosophy framework docs from ~/Documents/Research/
if [[ -d "$HOME/Documents/Research/Deleuzean AI" ]]; then
    for f in "$HOME/Documents/Research/Deleuzean AI"/*; do
        [[ -f "$f" ]] && safe_copy "$f" "$GRADIENCE_II/archive/philosophy"
    done
else
    log_warn "Documents/Research/Deleuzean AI directory not found"
fi

echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "═══════════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════════"
echo "  Files copied:  $COPIED"
echo "  Files skipped: $SKIPPED"
echo "  Errors:        $ERRORS"
echo

if $DRY_RUN; then
    echo "  This was a DRY RUN. No files were actually copied."
    echo "  Run without --dry-run to perform the consolidation."
else
    echo "  Consolidation complete. Originals are still in place."
    echo "  Next steps:"
    echo "    1. Spot-check files in $GRADIENCE_II"
    echo "    2. When satisfied, delete originals from ~/Downloads etc."
    echo "    3. Consider adding the new archive/ directory to .gitignore"
    echo "       if you don't want large archival files in version control."
fi
echo
