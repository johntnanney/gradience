# Golden Downstream Script Patterns

This document demonstrates the recommended approaches for consuming Gradience audit JSON files in downstream applications like Bench.

## 🎯 Core Strategy

**Intelligent Fallback Approach:**
1. **Primary**: Use flagged high-impact layers (best signal)
2. **Fallback**: Use top-N layers by priority_score (good signal)  
3. **Ultimate Fallback**: Use layers with disagreement (weak signal)

## 📋 Pattern 1: Simple Pattern (User Suggested)

**Best for**: Quick scripts, direct integration, minimal dependencies

```python
import json

audit = json.load(open("audit_results.json"))
pd = audit["policy_disagreement_analysis"]

focus = pd.get("flagged_layers", [])
# Optional: fall back to top-N by priority_score if nothing flagged
if not focus:
    all_layers = pd.get("all_layers_with_disagreement", [])  
    focus = sorted(all_layers, key=lambda x: x["flagging_rationale"].get("priority_score", 0), reverse=True)[:3]

for layer in focus:
    r = layer["flagging_rationale"]
    print(
        f"Validate {layer['layer_name']}: "
        f"spread={r['spread']} (≥{r.get('spread_threshold','?')}), "
        f"uniform_mult={r['uniform_mult']:.2f} (≥{r.get('uniform_mult_threshold','?')}), "
        f"share={r.get('importance_share',0):.3f}"
    )
```

**Example Output:**
```
Validate transformer.h.0.self_attn.q_proj: spread=6 (≥4.0), uniform_mult=2.08 (≥1.5), share=0.694
```

## 🛠️ Pattern 2: Comprehensive Pattern

**Best for**: Production systems, complex validation logic, detailed analysis

Features:
- Robust error handling
- Priority categorization (HIGH/MEDIUM/LOW/SKIP)
- Bench configuration generation
- Detailed threshold analysis
- Multiple output formats

```bash
# Basic usage
python3 golden_downstream_pattern.py audit_results.json

# With fallback control
python3 golden_downstream_pattern.py audit_results.json --top-n 5

# Generate Bench config
python3 golden_downstream_pattern.py audit_results.json --generate-config
```

**Example Output:**
```
🎯 Using 1 flagged high-impact layers

📊 LAYER ANALYSIS (1 layers):
1. transformer.h.0.self_attn.q_proj
   Status: HIGH PRIORITY (Flagged as high-impact by all criteria)
   Metrics: spread=6.000 (≥4.000) ✅, uniform_mult=2.083 (≥1.500) ✅
   Share: 0.694, Priority: 3.12

🚀 VALIDATION RECOMMENDATIONS:
🔥 HIGH PRIORITY (1 layers):
   • transformer.h.0.self_attn.q_proj: Use per-layer optimization
```

## 🔧 Integration Examples

### Bench Protocol Integration

```python
def create_focused_compression_configs(audit_data):
    """Integrate with Bench using focus set directly."""
    
    pd = audit_data.get("policy_disagreement_analysis", {})
    focus_set = pd.get("disagreement_focus_set", {})
    per_layer_suggestions = audit_data.get("per_layer_suggestions", {})
    
    configs = []
    
    # Always include uniform baseline
    configs.append({
        "variant": "uniform", 
        "rank": 8,
        "description": "Conservative uniform baseline"
    })
    
    # Add focused per-layer if justified
    high_impact_layers = focus_set.get("high_impact_layers", [])
    if high_impact_layers and per_layer_suggestions:
        full_pattern = per_layer_suggestions.get("rank_pattern", {})
        focused_pattern = {
            layer: full_pattern[layer] 
            for layer in high_impact_layers 
            if layer in full_pattern
        }
        
        if focused_pattern:
            configs.append({
                "variant": "per_layer",
                "rank_pattern": focused_pattern,
                "focus_justification": focus_set.get("message"),
                "layers_tested": len(focused_pattern),
                "layers_skipped": len(full_pattern) - len(focused_pattern)
            })
    
    return configs
```

### Priority-Based Validation

```python
def prioritize_validation_layers(audit_data, max_layers=3):
    """Get validation priorities using golden pattern."""
    
    pd = audit_data["policy_disagreement_analysis"]
    
    # Strategy 1: Flagged layers (highest confidence)
    flagged = pd.get("flagged_layers", [])
    if flagged:
        return {"source": "flagged", "layers": flagged, "confidence": "high"}
    
    # Strategy 2: Top-N by priority score (medium confidence)
    all_layers = pd.get("all_layers_with_disagreement", [])
    if all_layers:
        by_priority = sorted(
            all_layers,
            key=lambda x: x["flagging_rationale"].get("priority_score", 0),
            reverse=True
        )
        top_layers = by_priority[:max_layers]
        return {"source": "priority_score", "layers": top_layers, "confidence": "medium"}
    
    # Strategy 3: No good signal
    return {"source": "none", "layers": [], "confidence": "none"}
```

## 📊 Usage Scenarios

### Scenario 1: High-Impact Model (Clear Signal)
- **Input**: Large transformer with clear energy hierarchy
- **Result**: 1-3 flagged layers with high importance shares
- **Action**: Focus Bench validation on flagged layers only
- **Benefit**: 70%+ validation complexity reduction

### Scenario 2: Flat Distribution (Weak Signal)  
- **Input**: Model with uniform energy distribution
- **Result**: No flagged layers, moderate priority scores
- **Action**: Use top-N by priority_score or skip per-layer entirely
- **Benefit**: Avoid wasted validation effort

### Scenario 3: Edge Case Handling
- **Input**: Malformed JSON, missing fields, no disagreement
- **Result**: Graceful fallbacks and clear error messages  
- **Action**: Appropriate fallback strategy or uniform-only validation
- **Benefit**: Robust production deployment

## 🎯 Key Benefits

### Simple Pattern Benefits:
- **Minimal code**: 15 lines for core logic
- **Direct integration**: Easy to embed in existing systems
- **User-suggested format**: Exactly as requested
- **Clear output**: Human-readable threshold analysis

### Comprehensive Pattern Benefits:
- **Production-ready**: Error handling, logging, validation
- **Multiple output modes**: Human-readable + machine-readable
- **Bench integration**: Direct configuration generation
- **Extensible**: Easy to add new validation strategies

## 🚀 Best Practices

1. **Always check flagged layers first** - highest signal quality
2. **Use priority_score for fallback** - better than arbitrary thresholds  
3. **Handle missing data gracefully** - robust error handling prevents crashes
4. **Preserve audit trail** - track which strategy was used and why
5. **Start conservative** - use max(k_values) for rank selection
6. **Focus on high-impact layers** - don't waste cycles on unimportant layers

## 📝 File Summary

- `simple_downstream_pattern.py`: Minimal 15-line implementation
- `golden_downstream_pattern.py`: Full-featured production implementation  
- `test_fallback_scenario.py`: Creates test data for fallback scenarios
- `sample_audit_for_explain.json`: Sample audit with flagged layers
- `fallback_scenario_audit.json`: Sample audit with no flagged layers

Both patterns implement the same core strategy but with different levels of sophistication. Choose based on your integration needs and error handling requirements.