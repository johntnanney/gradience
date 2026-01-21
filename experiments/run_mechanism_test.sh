#!/bin/bash
# Mechanism Testing Experiment Runner
# 
# Runs the controlled experiment to test whether audit-guided per-layer 
# rank patterns provide real benefit beyond simple heterogeneity through
# comparison with shuffled control variants.

set -euo pipefail

# Configuration
CONFIG_FILE="experiments/mechanism_test_config.yaml"
OUTPUT_BASE="experiments/mechanism_test_results"
SEEDS=(42 43 44)

# Validate environment
echo "=== Mechanism Test Experiment Runner ==="
echo "Validating environment..."

# Check if config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check CUDA availability
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "⚠️  Warning: CUDA not available - this will be slow"
fi

# Check model access (HuggingFace login)
echo "Checking model access..."
if ! python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')" > /dev/null 2>&1; then
    echo "❌ Error: Cannot access Mistral-7B model. Please ensure:"
    echo "  1. HuggingFace CLI is logged in: huggingface-cli login"
    echo "  2. You have access to mistralai/Mistral-7B-v0.1"
    exit 1
fi

echo "✓ Environment validation passed"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Run experiments for each seed
echo "=== Starting Multi-Seed Runs ==="
for seed in "${SEEDS[@]}"; do
    echo "--- Running experiment with seed $seed ---"
    
    # Create seed-specific output directory
    seed_output="$OUTPUT_BASE/seed_${seed}"
    mkdir -p "$seed_output"
    
    # Update config with current seed
    sed "s/seed: .*/seed: $seed/" "$CONFIG_FILE" > "${seed_output}/config_seed${seed}.yaml"
    
    # Run the benchmark
    echo "Starting gradience bench for seed $seed..."
    python3 -m gradience.cli bench \
        --config "${seed_output}/config_seed${seed}.yaml" \
        --output "$seed_output" \
        --verbose \
        2>&1 | tee "${seed_output}/run_log.txt"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✓ Seed $seed completed successfully"
    else
        echo "❌ Seed $seed failed - check ${seed_output}/run_log.txt"
        exit 1
    fi
done

echo "=== All Runs Completed ==="

# Aggregate results across seeds
echo "--- Aggregating multi-seed results ---"
python3 -c "
import sys
sys.path.append('.')
from gradience.bench.protocol import create_multi_seed_aggregated_report
import json
from pathlib import Path

# Load individual reports
reports = []
for seed in [42, 43, 44]:
    report_path = Path('$OUTPUT_BASE') / f'seed_{seed}' / 'bench.json'
    if report_path.exists():
        with open(report_path) as f:
            reports.append(json.load(f))

# Create aggregate
if reports:
    config = reports[0].get('config', {})
    aggregate = create_multi_seed_aggregated_report(
        reports, config, Path('$OUTPUT_BASE')
    )
    
    # Save aggregated results
    with open('$OUTPUT_BASE/aggregated_results.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
        
    print('✓ Aggregated results saved to $OUTPUT_BASE/aggregated_results.json')
    
    # Print summary
    probe_acc = aggregate.get('probe', {}).get('accuracy', {}).get('mean', 0)
    compressed = aggregate.get('compressed', {})
    
    print(f'\\n=== EXPERIMENT SUMMARY ===')
    print(f'Probe baseline: {probe_acc:.3f}')
    
    for variant in ['per_layer', 'per_layer_shuffled']:
        if variant in compressed:
            acc = compressed[variant].get('accuracy', {}).get('mean', 0)
            delta = compressed[variant].get('delta_vs_probe', {}).get('mean', 0)
            print(f'{variant}: {acc:.3f} (Δ{delta:+.3f})')
    
    # Mechanism test result
    if 'per_layer' in compressed and 'per_layer_shuffled' in compressed:
        per_layer_acc = compressed['per_layer']['accuracy']['mean']
        shuffled_acc = compressed['per_layer_shuffled']['accuracy']['mean']
        mechanism_benefit = per_layer_acc - shuffled_acc
        
        print(f'\\nMechanism test result:')
        print(f'  per_layer advantage: {mechanism_benefit:+.3f}')
        
        if mechanism_benefit > 0.01:  # 1% threshold
            print(f'  ✓ SIGNIFICANT: Audit-guided placement provides real benefit')
        else:
            print(f'  ❌ NOT SIGNIFICANT: No clear mechanism benefit detected')
else:
    print('❌ No valid reports found for aggregation')
"

echo "=== Mechanism Test Complete ==="
echo "Results available in: $OUTPUT_BASE/"
echo "- Individual runs: $OUTPUT_BASE/seed_*/bench.json"
echo "- Aggregated analysis: $OUTPUT_BASE/aggregated_results.json"