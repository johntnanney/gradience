# Bench Policies

Machine-consumable policy artifacts for automated compliance checking.

## Safe Uniform Baseline Policy

**File**: `safe_uniform.yaml`  
**Scope**: DistilBERT/SST-2 validation  
**Status**: Policy v0.1 (Screening+ level validation)

### Usage Example

```python
import yaml
from pathlib import Path

# Load policy
policy_path = Path("gradience/bench/policies/safe_uniform.yaml")
with open(policy_path) as f:
    policy = yaml.safe_load(f)

# Check compliance
def check_uniform_safety(rank, model_family="distilbert", task="glue/sst2"):
    if policy["scope"]["model_family"] != model_family:
        return {"compliant": False, "reason": "Model family not covered by policy"}
    
    if policy["scope"]["task"] != task:
        return {"compliant": False, "reason": "Task not covered by policy"}
    
    primary_rank = policy["recommendations"]["primary_uniform_rank"]
    conservative_rank = policy["recommendations"]["conservative_uniform_rank"] 
    avoid_ranks = policy["recommendations"]["avoid_uniform_ranks"]
    
    if rank in avoid_ranks:
        return {"compliant": False, "reason": f"Rank {rank} violates safety policy"}
    elif rank == primary_rank:
        return {"compliant": True, "level": "primary", "compression": "25.0%"}
    elif rank == conservative_rank:
        return {"compliant": True, "level": "conservative", "compression": "16.6%"}
    else:
        return {"compliant": "unknown", "reason": f"Rank {rank} not validated by policy"}

# Example usage
print(check_uniform_safety(20))  # {'compliant': True, 'level': 'primary', 'compression': '25.0%'}
print(check_uniform_safety(16))  # {'compliant': False, 'reason': 'Rank 16 violates safety policy'}
```

### Integration Points

- **`gradience check`**: Policy compliance validation
- **`gradience audit`**: Automated safety recommendations  
- **`gradience.bench`**: Policy-driven baseline selection
- **CI/CD pipelines**: Automated safety gates

### Policy Versioning

Policies are versioned to enable:
- Backward compatibility tracking
- Policy evolution over time  
- Validation level upgrades (Screening → Screening+ → Certifiable)
- Multi-task/model policy aggregation

### Regression Testing

Policy decisions are protected by regression tests:

```bash
# Run policy regression tests
./scripts/test_policy_regression.sh

# Or run directly with pytest
python -m pytest tests/test_bench/test_policy_regression.py -v
```

These tests guard against accidental policy drift by asserting:
- r=20 remains primary safe uniform baseline
- r=16 remains marked unsafe (empirically proven 0% pass rate)
- Safety criteria thresholds (67% pass rate, -2.5% worst delta) unchanged
- Empirical evidence and limitations properly documented