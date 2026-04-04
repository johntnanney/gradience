# Research Docs

**Audience:** researcher, reviewer, deep collaborator
**Status:** curated entry to active sidecar record
**Purpose:** summarize findings, settled/open status, and evidence links
**Canonical for:** research-facing documentation index
**Supersedes:** direct reliance on raw sidecar note listings as first entrypoint
**See also:** [`../architecture/README.md`](../architecture/README.md)

## How these layers relate

The research documentation exists at three levels of detail. They cover the same findings but serve different purposes:

1. **[Technical Report](../technical-report.md)** (Sections 2, 3, 6, 7) — the primary synthesis. Presents the full argument: why spectral geometry carries compatibility information, the conjunctive failure mechanism, ruled-out alternatives, and open frontiers. **Start here.** If you read one document about the research, read this.

2. **Canonical summaries** (below) — deep dives on specific topics. These expand on individual findings that the technical report covers in compressed form. Use them when the report gives you the conclusion and you want the full evidence trail.

3. **Sidecar archive** (`sidecar/`) — the raw research record. 121 notes, 126 structured data outputs, 69 figures. This is the complete record of every hypothesis tested, every analysis run, every negative result. Navigable via the packet indexes below but not intended as a first entry point.

The technical report is the maintained synthesis. The summaries and sidecar archive are reference material — they will not be updated to track editorial changes in the report, but the underlying findings they document remain valid.

## Start here

- [`../technical-report.md`](../technical-report.md) — **the end-to-end argument**: theory, mechanism, field trials, and open frontiers
- [`../field-trial-retrospective.md`](../field-trial-retrospective.md) — what we expected, what surprised us, and what changed (companion to the report's §5)
- [`research-overview.md`](research-overview.md) — orientation to the two research lines (mechanism + Route 2)
- [`route2-summary.md`](route2-summary.md) — decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge

## Core research set

- Technical report: [`../technical-report.md`](../technical-report.md)
- Theory document: [`../THEORY.md`](../THEORY.md)
- Findings overview: [`../FINDINGS.md`](../FINDINGS.md)
- Bounded/experimental feature matrix: [`../00_start_here/stable-vs-experimental.md`](../00_start_here/stable-vs-experimental.md)

## Canonical summaries (deep dives)

These expand on specific topics covered more briefly in the technical report:

- [`summaries/where-the-research-stands.md`](summaries/where-the-research-stands.md) — expands Technical Report §3 (mechanism ladder, conjunctive model, behavioral signatures)
- [`summaries/settled-open-next.md`](summaries/settled-open-next.md) — expands Technical Report §7 (16 settled claims, 6 open questions, prioritized next steps)
- [`summaries/ruled-out-mechanisms.md`](summaries/ruled-out-mechanisms.md) — expands Technical Report §6 (8 primary + 5 ancillary eliminations with full evidence)
- [`summaries/cross-artifact-product-relevance-summary.md`](summaries/cross-artifact-product-relevance-summary.md) — expands Technical Report §7.4 (LoHa, checkpoint delta, routing portability)
- [`summaries/aggregation-sensitive-route2-summary.md`](summaries/aggregation-sensitive-route2-summary.md) — expands Technical Report §7.4 (aggregation as computational seam, 12-case panel)
- [`summaries/behavioral-route2-summary.md`](summaries/behavioral-route2-summary.md) — expands Technical Report §3.5 (behavioral signatures, 4,000-example analysis)
- [`summaries/README.md`](summaries/README.md) — index to all summaries

## Packets (self-contained research bundles)

- [`packet/README.md`](packet/README.md) — main research packet (mechanism ladder + evidence table + ruled-out + GPU reentry)
- [`packet/route2/00_route2_packet_index.md`](packet/route2/00_route2_packet_index.md) — Route 2 packet (cross-artifact + aggregation + behavioral bridge)
