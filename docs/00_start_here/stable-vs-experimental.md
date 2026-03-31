# Stable vs Experimental Matrix

**Audience:** practitioner, maintainer, collaborator  
**Status:** stable  
**Purpose:** explicit capability/status matrix  
**Canonical for:** scope and confidence boundaries  
**Supersedes:** informal status spread across multiple memos  
**See also:** [`project-map.md`](project-map.md)

| Capability / Program | Scope | Status | Primary Audience | Evidence Level | User-Facing | Canonical Docs |
|---|---|---|---|---|---|---|
| Adapter merge preflight | LoRA adapter inventories | stable | practitioner | field-validated (5 inventories, 53+ pairs) | yes | [`../product/product-validation.md`](../product/product-validation.md) |
| Same-family routing inside preflight | small-encoder classification families | stable (bounded) | practitioner | targeted confirmation (T01) | yes | [`../product/product-validation.md`](../product/product-validation.md) |
| Near-miss severity ordering | preflight action plan | stable (bounded) | practitioner | targeted confirmation (T02) | yes | [`../product/product-validation.md`](../product/product-validation.md) |
| Checkpoint triage alpha | shared-base checkpoint inventories | alpha | practitioner/maintainer | field trial T02 + Ring 2 stages | yes (alpha) | [`../product/checkpoint-triage-alpha-workflow.md`](../product/checkpoint-triage-alpha-workflow.md) |
| Routing pilot | adapter routing/confusability | experimental | maintainer/researcher | single validated pilot | no | [`../research/route2-summary.md`](../research/route2-summary.md) |
| Ring 1 (LoHa generalization) | low-rank PEFT artifact class | experimental (validated) | maintainer/researcher | completed ring evaluation | no | [`../architecture/broadened-substrate-scope.md`](../architecture/broadened-substrate-scope.md) |
| Ring 2 (checkpoint delta representation path) | summary-based checkpoint delta triage | experimental (validated) | maintainer/researcher | completed ring evaluation | no | [`../architecture/broadened-substrate-scope.md`](../architecture/broadened-substrate-scope.md) |
| Cross-artifact compatibility portability | LoRA/LoHa/checkpoint analysis portability | research (settled in bounded scope) | researcher | sidecar program + stability substudy | no | [`../research/summaries/cross-artifact-product-relevance-summary.md`](../research/summaries/cross-artifact-product-relevance-summary.md) |
| Aggregation-sensitive compatibility | merge/routing/triage aggregation families | research (settled core + guarded thresholds) | researcher | sidecar program + stability substudy + mixed-evidence pass | no | [`../research/summaries/aggregation-sensitive-route2-summary.md`](../research/summaries/aggregation-sensitive-route2-summary.md) |
| Behavioral Route 2 bridge | behavioral grounding of Route 2 profiles | research (settled bounded) | researcher | sidecar behavioral bridge | no | [`../research/summaries/behavioral-route2-summary.md`](../research/summaries/behavioral-route2-summary.md) |
| DeBERTa adjudication work | mechanism portability to third backbone | gpu-blocked | researcher | pre-registered protocol only | no | [`../research/summaries/settled-open-next.md`](../research/summaries/settled-open-next.md) |
