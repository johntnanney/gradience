#!/bin/bash
#
# Demo script for LoRA gain audit functionality (v0.7.0)
# 
# Runs the fastest CPU bench configuration and extracts key gain metrics
# from the audit results to demonstrate the new functionality.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}🔍 LoRA Gain Audit Demo (v0.7.0)${NC}"
echo "==============================="
echo

# Set output directory
OUT="/tmp/gradience_gain_audit_demo"
rm -rf "$OUT"

echo -e "${YELLOW}📊 Running DistilBERT SST2 mini bench (smoke mode)...${NC}"
echo "Config: gradience/bench/configs/distilbert_sst2_mini_validation.yaml"
echo "Output: $OUT"
echo

# Run the bench in smoke mode for fastest execution
PYTHONPATH=/Users/john/code/gradience python3 -m gradience.bench.run_bench \
    --config gradience/bench/configs/distilbert_sst2_mini_validation.yaml \
    --output "$OUT" \
    --smoke > /tmp/demo_gain_audit.log 2>&1

# Check if bench completed successfully
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Bench run failed. Check /tmp/demo_gain_audit.log for details.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Bench completed successfully!${NC}"
echo

# Extract key metrics from audit.json
AUDIT_FILE="$OUT/probe_r16/audit.json"

if [ ! -f "$AUDIT_FILE" ]; then
    echo -e "${RED}❌ Audit file not found: $AUDIT_FILE${NC}"
    exit 1
fi

echo -e "${BOLD}${BLUE}📈 Gain Audit Results${NC}"
echo "===================="
echo

# Extract magnitude metrics using jq
echo -e "${BOLD}Update Magnitude:${NC}"
MEAN_FRO=$(python3 -c "
import json
with open('$AUDIT_FILE') as f:
    data = json.load(f)
    fro_mean = data.get('summary', {}).get('gain', {}).get('delta_fro_mean', 0)
    print(f'{fro_mean:.6f}')
")

MEAN_OP=$(python3 -c "
import json
with open('$AUDIT_FILE') as f:
    data = json.load(f)
    op_mean = data.get('summary', {}).get('gain', {}).get('delta_op_mean', 0) 
    print(f'{op_mean:.6f}')
")

echo "• Mean ||ΔW||_F: $MEAN_FRO"
echo "• Mean ||ΔW||_2: $MEAN_OP"
echo

# Extract top 5 layers by energy
echo -e "${BOLD}Top 5 Layers by Δ Energy:${NC}"
python3 -c "
import json
with open('$AUDIT_FILE') as f:
    data = json.load(f)
    composition = data.get('composition', {})
    layers = composition.get('top_k', {}).get('layers', [])
    
    for i, layer_info in enumerate(layers[:5], 1):
        layer_num = layer_info['layer']
        share = layer_info['share']
        energy = layer_info['energy_fro2']
        print(f'{i}. Layer {layer_num}: {share:.1%} ({energy:.6f})')
"
echo

# Extract energy concentration summary
echo -e "${BOLD}Energy Concentration:${NC}"
python3 -c "
import json
with open('$AUDIT_FILE') as f:
    data = json.load(f)
    composition = data.get('composition', {})
    
    # HHI concentration index
    hhi = composition.get('concentration_index', 0)
    
    # Top-10% share
    top_10pct = composition.get('top_10pct', {})
    share = top_10pct.get('share', 0)
    n_layers = top_10pct.get('n', 0)
    
    print(f'• Top-{n_layers} layers (10%): {share:.1%} of energy')
    print(f'• Concentration index (HHI): {hhi:.3f}')
    
    # Interpretation
    if hhi > 0.4:
        print('• 🚨 Highly concentrated adaptation')
    elif hhi > 0.25:
        print('• ⚠️  Moderately concentrated adaptation') 
    else:
        print('• ✅ Well distributed adaptation')
"
echo

# Extract compression results summary  
echo -e "${BOLD}Compression Summary:${NC}"
BENCH_FILE="$OUT/bench.md"

if [ -f "$BENCH_FILE" ]; then
    BEST_COMPRESSION=$(grep "Best compression:" "$BENCH_FILE" | sed 's/.*Best compression:[[:space:]]*//')
    if [ -n "$BEST_COMPRESSION" ]; then
        echo "• Best compression variant: $BEST_COMPRESSION"
        
        # Show results table header for context
        echo "• Results from compression variants:"
        grep -A5 "| Variant |" "$BENCH_FILE" | grep -E "(Variant|per_layer)" | head -3
    fi
fi
echo

echo -e "${BOLD}${GREEN}🎉 Demo completed successfully!${NC}"
echo
echo -e "${YELLOW}📁 Full results available in:${NC}"
echo "• Audit data: $AUDIT_FILE"
echo "• Human report: $OUT/bench.md" 
echo "• Machine report: $OUT/bench.json"
echo
echo -e "${BLUE}ℹ️  This demo showcased the comprehensive LoRA gain audit functionality${NC}"
echo -e "${BLUE}   including magnitude metrics, energy concentration analysis, and${NC}"
echo -e "${BLUE}   composition-based insights for LoRA adapter optimization.${NC}"