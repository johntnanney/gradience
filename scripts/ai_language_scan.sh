#!/usr/bin/env bash
# =============================================================================
# ai_language_scan.sh — pre-submission AI-language audit
#
# Scans a document for the lexical, phrasal, and structural patterns
# documented in AI_LANGUAGE_SCAN_CHECKLIST.md. Reports hits per layer and
# a summary cluster score.
#
# Usage:
#   ./scripts/ai_language_scan.sh <file>
#   ./scripts/ai_language_scan.sh papers/n134_workshop/draft_v2_thesis_b.tex
#
# The script reports hits; it does not modify the file. Domain-calibrate the
# output: some hits are legitimate in the document's context (e.g., "robust"
# in a statistics paper). Cluster density is the diagnostic signal, not
# individual hits.
#
# Exit code: 0 always. The script is informational; the user judges severity.
# =============================================================================

set -uo pipefail

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <file>

Scans <file> for AI-language tells documented in AI_LANGUAGE_SCAN_CHECKLIST.md.
Reports per-layer hits and a summary cluster score.

Examples:
  $0 papers/n134_workshop/draft_v2_thesis_b.tex
  $0 papers/benchmark_reliability_study/manuscript/draft_v1.tex

For each layer, output is:
  <layer name>: <total hits>
  <line>:<matched text>     [up to 5 example hits per layer]

A "cluster score" at the end aggregates hit density across layers. Treat
scores >= 6 as a strong AI-tell signal worth reviewing; lower scores
warrant a glance but rarely action.
EOF
    exit 0
fi

FILE="$1"

if [[ ! -f "$FILE" ]]; then
    echo "Error: $FILE not found" >&2
    exit 1
fi

# Color setup (skip if not a tty)
if [[ -t 1 ]]; then
    HEADER='\033[1;36m'
    HIT='\033[0;33m'
    OK='\033[0;32m'
    DIM='\033[0;90m'
    RESET='\033[0m'
else
    HEADER=''
    HIT=''
    OK=''
    DIM=''
    RESET=''
fi

# Total hits accumulator
TOTAL_HITS=0

run_layer() {
    local layer_name="$1"
    local pattern="$2"
    local case_flag="${3:-i}"
    local description="${4:-}"

    local hits
    if [[ "$case_flag" == "i" ]]; then
        hits=$(grep -niE "$pattern" "$FILE" 2>/dev/null || true)
    else
        hits=$(grep -nE "$pattern" "$FILE" 2>/dev/null || true)
    fi

    local count
    if [[ -z "$hits" ]]; then
        count=0
    else
        count=$(echo "$hits" | wc -l | tr -d ' ')
    fi

    printf "${HEADER}%s${RESET}" "$layer_name"
    if [[ -n "$description" ]]; then
        printf " ${DIM}(%s)${RESET}" "$description"
    fi
    printf ": "

    if [[ $count -eq 0 ]]; then
        printf "${OK}0 hits${RESET}\n"
    else
        printf "${HIT}%d hits${RESET}\n" "$count"
        echo "$hits" | head -5 | sed "s/^/  /"
        if [[ $count -gt 5 ]]; then
            printf "  ${DIM}... and %d more${RESET}\n" $((count - 5))
        fi
    fi
    echo

    TOTAL_HITS=$((TOTAL_HITS + count))
}

echo
printf "${HEADER}=== AI-language scan: $FILE ===${RESET}\n\n"

# -----------------------------------------------------------------------------
# Layer 1a — High-precision overused vocabulary
# Words with little legitimate technical use; these are unambiguous AI tells
# in academic prose. Contributes to cluster score at full weight.
# -----------------------------------------------------------------------------

run_layer "Layer 1a: high-precision AI-tell vocabulary" \
    '\b(delve|delves|delving|delved|underscore|underscores|underscoring|underscored|paramount|tapestry|scaffold|scaffolds|scaffolding|ascertain|ascertains|ascertained|showcas|pivotal|realm|meticulous|illuminate|illuminates|illuminated|unveil|unveiled|unveiling|elevate|elevates|elevated)\b' \
    "i" "delve, underscore, paramount, tapestry, etc."

# -----------------------------------------------------------------------------
# Layer 1b — Domain-overlap candidates
# Words common in measurement/statistics/ML/psychometrics domains as well as
# in AI-flavored prose. Reports informationally but does NOT contribute to
# cluster score; user must judge per-hit whether usage is technical or AI tell.
# -----------------------------------------------------------------------------

LAYER_1B_HITS=$(grep -niE '\b(foster|fostered|fostering|fosters|harness|harnessed|cultivate|cultivated|cultivating|empower|empowers|empowering|empowered|optimize|optimized|optimizing|streamline|streamlined|streamlining|encompass|encompasses|encompassed|embody|embodied|embodies|manifest|manifests|manifesting|manifested|traverse|traverses|traversed|traversing)\b' "$FILE" 2>/dev/null || true)
if [[ -z "$LAYER_1B_HITS" ]]; then
    LAYER_1B_COUNT=0
else
    LAYER_1B_COUNT=$(echo "$LAYER_1B_HITS" | wc -l | tr -d ' ')
fi

printf "${HEADER}Layer 1b: domain-overlap vocabulary${RESET} ${DIM}(harness, optimize, manifest, etc. — informational only)${RESET}: "
if [[ $LAYER_1B_COUNT -eq 0 ]]; then
    printf "${OK}0 hits${RESET}\n"
else
    printf "${HIT}%d hits${RESET} ${DIM}(does not contribute to cluster score; review for domain-legitimacy)${RESET}\n" "$LAYER_1B_COUNT"
    echo "$LAYER_1B_HITS" | head -5 | sed "s/^/  /"
    if [[ $LAYER_1B_COUNT -gt 5 ]]; then
        printf "  ${DIM}... and %d more${RESET}\n" $((LAYER_1B_COUNT - 5))
    fi
fi
echo

# -----------------------------------------------------------------------------
# Layer 2 — Hype vocabulary
# -----------------------------------------------------------------------------

run_layer "Layer 2: hype vocabulary" \
    '\b(game-changer|revolutioniz|groundbreaking|cutting-edge|state-of-the-art|paradigm shift|seismic shift|unprecedented|disruptive|next-generation|transformative)\b' \
    "i" "game-changer, groundbreaking, etc."

# -----------------------------------------------------------------------------
# Layer 3 — Performative-engagement vocabulary
# -----------------------------------------------------------------------------

run_layer "Layer 3: performative-engagement vocabulary" \
    '\b(compelling|fascinating|intriguing|captivating|riveting|enthralling|profound|profoundly)\b' \
    "i" "compelling, fascinating, profound, etc."

# -----------------------------------------------------------------------------
# Layer 4 — Framing pivots
# -----------------------------------------------------------------------------

run_layer "Layer 4: framing pivots" \
    "this is where [a-z]+ comes in|the (beauty|elegance|power|magic) of [a-z]+|here'?s the (thing|catch|kicker|rub)|let'?s dive in|without further ado|when it comes to|in the realm of|in today'?s (world|landscape|era|age)|in the world of|picture this|imagine that|what (sets|makes) [a-z]+ (apart|unique)" \
    "i" "this is where X comes in, beauty of, etc."

# -----------------------------------------------------------------------------
# Layer 5 — Substitution patterns (verb replacements)
# -----------------------------------------------------------------------------

run_layer "Layer 5: substitution patterns" \
    "\b(serves|stands) as\b|\bplays a [a-z]+ role\b|speaks volumes|testament to|sheds light on|shed new light on|underscores the importance" \
    "i" "serves as, plays a role, sheds light on, etc."

# -----------------------------------------------------------------------------
# Layer 6 — Hedge / boilerplate qualifiers
# -----------------------------------------------------------------------------

run_layer "Layer 6: hedge / boilerplate qualifiers" \
    "it'?s? (worth|important) (to note|noting|mentioning)|\bit goes without saying\b|\bneedless to say\b|\bit can be argued\b|\bsome might say\b|\bone might say\b|\bit is (generally|widely) (accepted|held|believed)" \
    "i" "it's worth noting, it can be argued, etc."

# -----------------------------------------------------------------------------
# Layer 7 — AI-cadence rhetorical pivots
# -----------------------------------------------------------------------------

run_layer "Layer 7: AI-cadence rhetorical pivots" \
    "not (merely|simply|just) [a-z ]+\\.\\s+It is\b|isn'?t (just|merely|simply) [a-z]|not just [a-z]+ but [a-z]+|not only [a-z ]+ but also" \
    "i" "not merely X. It is Y; not just X but Y"

# -----------------------------------------------------------------------------
# Layer 8 — Verbosity substitutions
# -----------------------------------------------------------------------------

run_layer "Layer 8: verbosity substitutions" \
    "\b(utilize|utilizes|utilized|utilizing|facilitate|facilitates|facilitated|facilitating|elucidate|elucidates|elucidated|endeavor|endeavors|endeavored|endeavoring|commence|commences|commenced|commencing|terminate|terminates|terminated|terminating)\b|in order to|due to the fact that|in light of the fact that|owing to the fact that|for the purposes of|in spite of the fact that|with respect to|in the event that|at this point in time|at the present time" \
    "i" "utilize, facilitate, in order to, etc."

# -----------------------------------------------------------------------------
# Layer 9 — Academic-methodology boilerplate
# -----------------------------------------------------------------------------

run_layer "Layer 9: academic-methodology boilerplate" \
    "this (study|paper|work) (aims|seeks) to|the aim of this (study|paper|work) is|let us (examine|consider|explore)|a growing body of (evidence|literature|research)|recent advances have shown|future research could|more work is needed|this (study|paper|work) contributes to the literature|valuable insights|promising results" \
    "i" "this study aims to, growing body of evidence, etc."

# -----------------------------------------------------------------------------
# Layer 10 — Closing/opening boilerplate (conversational AI leaks)
# -----------------------------------------------------------------------------

run_layer "Layer 10: closing/opening boilerplate" \
    "i hope this helps|let me know if you have|feel free to reach out|looking forward to hearing|certainly!|absolutely!|great question|i'?d be happy to|that'?s a great point|let me break this down" \
    "i" "I hope this helps, certainly!, great question, etc."

# -----------------------------------------------------------------------------
# Layer 11 — Connective and adverb-opener overuse (count, don't just match)
# -----------------------------------------------------------------------------

printf "${HEADER}Layer 11: connective and adverb-opener counts${RESET} ${DIM}(individual rates may be fine; clustered = tell)${RESET}\n"
CONN_TOTAL=0
for opener in "Furthermore," "Moreover," "Additionally," "However," "Nevertheless," "Consequently," "Therefore," "Thus," "Hence," "In conclusion," "In summary," "Importantly," "Crucially," "Notably," "Significantly," "Indeed," "Specifically," "Particularly," "Especially," "Effectively," "Essentially," "Fundamentally," "Of course,"; do
    count=$(grep -c "^$opener" "$FILE" 2>/dev/null) || count=0
    if [[ $count -gt 0 ]]; then
        printf "  %-18s %d\n" "$opener" "$count"
    fi
    CONN_TOTAL=$((CONN_TOTAL + count))
done
printf "  ${DIM}total connective/opener hits: %d${RESET}\n" "$CONN_TOTAL"
TOTAL_HITS=$((TOTAL_HITS + CONN_TOTAL))
echo

# -----------------------------------------------------------------------------
# Layer 12 — Em-dash density
# -----------------------------------------------------------------------------

EM_DASH_COUNT=$(grep -o -- '---' "$FILE" 2>/dev/null | wc -l | tr -d ' ')
PAGE_EST=$(grep -c '^' "$FILE" | awk '{ printf "%d", $1 / 50 }')   # rough page estimate; ~50 source lines per page
if [[ $PAGE_EST -lt 1 ]]; then PAGE_EST=1; fi
EM_DASH_PER_PAGE=$((EM_DASH_COUNT / PAGE_EST))

printf "${HEADER}Layer 12: em-dash density${RESET} ${DIM}(>=6 per page is a tell)${RESET}\n"
printf "  total em-dashes (---): %d\n" "$EM_DASH_COUNT"
printf "  rough source-page estimate: %d\n" "$PAGE_EST"
printf "  em-dashes per page (est.): %d" "$EM_DASH_PER_PAGE"
if [[ $EM_DASH_PER_PAGE -ge 6 ]]; then
    printf " ${HIT}(elevated)${RESET}\n"
    TOTAL_HITS=$((TOTAL_HITS + 5))   # treat elevated em-dash as 5 hits in cluster score
else
    printf " ${OK}(normal)${RESET}\n"
fi
echo

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

printf "${HEADER}=== Summary ===${RESET}\n"
printf "  total cluster score: %d\n" "$TOTAL_HITS"
if [[ $TOTAL_HITS -ge 12 ]]; then
    printf "  ${HIT}interpretation:${RESET} strong AI-tell signal; review recommended\n"
elif [[ $TOTAL_HITS -ge 6 ]]; then
    printf "  ${HIT}interpretation:${RESET} moderate AI-tell signal; review hits above for clustering\n"
elif [[ $TOTAL_HITS -ge 3 ]]; then
    printf "  ${DIM}interpretation:${RESET} mild signal; individual hits worth a glance, document overall likely fine\n"
else
    printf "  ${OK}interpretation:${RESET} clean; document passes the lexical and phrasal scan\n"
fi
echo
printf "${DIM}Note: this scan covers lexical and phrasal patterns. Stylometric${RESET}\n"
printf "${DIM}signals (sentence-length variance, paragraph uniformity, register${RESET}\n"
printf "${DIM}consistency) require human review and aren't captured here. See${RESET}\n"
printf "${DIM}AI_LANGUAGE_SCAN_CHECKLIST.md §8 for those.${RESET}\n"
