# Policy Scoreboard Design

## Vision: Self-Improving Policy Framework

Transform "policies as hypotheses" into "policies with track records" by tracking performance across benchmark runs.

## Key Metrics to Track

For each policy suggestion, measure:

### 1. Pass Rate
- **Definition**: How often policy passes Bench validation 
- **Calculation**: `passes / total_attempts`
- **Details**: Pass = worst-seed Δ ≥ performance threshold

### 2. Optimality Score  
- **Definition**: How close to best-performing rank among candidates
- **Calculation**: `1.0 - (best_performance - policy_performance) / best_performance`
- **Range**: [0.0, 1.0] where 1.0 = optimal

### 3. Conservatism Bias
- **Definition**: Tendency to be too aggressive/conservative
- **Calculation**: `(policy_rank - optimal_rank) / optimal_rank`
- **Interpretation**: 
  - Negative = too aggressive (ranks too low)
  - Positive = too conservative (ranks too high)

### 4. Reliability Score
- **Definition**: Consistency across different models/tasks
- **Calculation**: `1.0 - std_dev(optimality_scores)`
- **Range**: [0.0, 1.0] where 1.0 = perfectly consistent

## Data Structure

```json
{
  "policy_scoreboard_version": "1.0",
  "generated_at": "2026-01-28T10:30:00Z",
  "total_benchmarks": 15,
  "policies": {
    "energy_90": {
      "total_attempts": 15,
      "passes": 12,
      "pass_rate": 0.80,
      "optimality_scores": [0.95, 0.87, 0.92, ...],
      "avg_optimality": 0.91,
      "conservatism_bias": 0.05,
      "reliability_score": 0.88,
      "trend": "improving",
      "last_5_performance": [0.89, 0.92, 0.94, 0.91, 0.95],
      "benchmarks": [
        {
          "config": "distilbert_sst2", 
          "date": "2026-01-28",
          "suggested_rank": 8,
          "actual_rank": 8, 
          "optimal_rank": 6,
          "pass": true,
          "optimality": 0.95,
          "performance_delta": 0.02
        }
      ]
    },
    "knee": {
      "total_attempts": 15,
      "passes": 10,
      "pass_rate": 0.67,
      "avg_optimality": 0.78,
      "conservatism_bias": -0.15,
      "reliability_score": 0.72,
      "trend": "stable",
      "note": "Tends to be aggressive - finds elbows early"
    }
  },
  "summary": {
    "best_overall_policy": "energy_90",
    "most_reliable": "energy_90", 
    "most_aggressive": "knee",
    "most_conservative": "oht",
    "recommendations": [
      "energy_90 shows consistent high performance (91% optimality)",
      "knee may be too aggressive (-15% bias) but good for aggressive compression",
      "oht very conservative, good for safety-critical applications"
    ]
  }
}
```

## Markdown Report Table

```markdown
## 📊 Policy Scoreboard

| Policy | Pass Rate | Optimality | Bias | Reliability | Trend | Notes |
|--------|-----------|------------|------|-------------|--------|-------|
| energy_90 ⭐ | 80% (12/15) | 91% | +5% 🔒 | 88% | ↗️ | Consistent high performer |
| knee | 67% (10/15) | 78% | -15% ⚡ | 72% | ➡️ | Aggressive, finds elbows early |
| erank | 73% (11/15) | 82% | -8% ⚡ | 79% | ↗️ | Good balance, improving |
| oht | 87% (13/15) | 85% | +22% 🔒 | 91% | ➡️ | Very conservative, reliable |

### Key Insights:
- **⭐ Best Overall**: energy_90 (91% optimality, 88% reliability)  
- **⚡ Most Aggressive**: knee (-15% bias - good for max compression)
- **🔒 Most Conservative**: oht (+22% bias - good for safety-critical)
- **📈 Improving**: energy_90, erank showing upward trends
```

## Implementation Plan

### Phase 1: Data Collection (during Bench runs)
1. Track policy suggestions and actual ranks used
2. Record performance results for each variant
3. Store metadata (model, task, date, seed)

### Phase 2: Scoreboard Computation (post-Bench)
1. Calculate metrics for each policy
2. Determine trends and biases
3. Generate insights and recommendations

### Phase 3: Output Generation
1. Create `policy_scoreboard.json` 
2. Add scoreboard table to aggregate reports
3. Update scoreboard across multiple benchmark runs

## File Locations

- **Storage**: `~/.gradience/policy_scoreboard.json` (persistent across runs)
- **Per-run**: `{output_dir}/policy_scoreboard_snapshot.json`
- **Reports**: Embedded in `aggregate_report.md`

## Benefits

1. **Evidence-Based Policy Selection**: Choose policies with proven track records
2. **Continuous Improvement**: Policies get better as more data is collected  
3. **Bias Awareness**: Understand when policies are too aggressive/conservative
4. **Model-Specific Insights**: See which policies work best for different model types
5. **Research Acceleration**: Quickly identify promising policy directions

## Example Usage Scenarios

### Practitioner
> "I need reliable compression for production. Policy scoreboard shows energy_90 has 91% optimality and 88% reliability - I'll use that."

### Researcher  
> "I'm exploring aggressive compression. Scoreboard shows knee has -15% bias (aggressive) but only 67% pass rate. Let me investigate why."

### Framework Developer
> "Scoreboard shows all policies trending downward on transformer models. Time to develop new transformer-specific policies."

This turns the policy system into a **learning, self-improving framework** rather than just a static collection of algorithms.