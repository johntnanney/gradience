#!/bin/bash
# Bench v0.1 Certifiable Standard Runner
# Executes the canonical 3-seed benchmark for public validation claims

set -e

echo "🏆 Bench v0.1 Certifiable Standard"
echo "=================================="
echo "Running canonical 3-seed benchmark (42, 123, 456)"
echo "Model/Task: DistilBERT + SST-2"
echo "Probe: r=32, Budget: 500 steps, Variants: per_layer + uniform_r20 + uniform_r24"
echo ""

# Track timing
start_time=$(date +%s)

# Function to run individual seeds
run_seed() {
    local seed=$1
    echo "🌱 Starting seed $seed..."
    
    python -m gradience.bench.run_bench \
        --config gradience/bench/configs/distilbert_sst2_certifiable_seed${seed}.yaml \
        --output bench_runs/cert_v0.1_seed${seed} \
        --ci
    
    echo "✅ Seed $seed completed"
    echo ""
}

# Option to run seeds in parallel or sequence
if [ "$1" = "--parallel" ]; then
    echo "🔀 Running seeds in parallel..."
    run_seed 42 &
    run_seed 123 &
    run_seed 456 &
    wait
    echo "✅ All parallel seeds completed"
else
    echo "🔗 Running seeds sequentially..."
    run_seed 42
    run_seed 123
    run_seed 456
    echo "✅ All sequential seeds completed"
fi

# Calculate timing
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "📊 CERTIFIABLE BENCHMARK COMPLETE"
echo "================================="
echo "Duration: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "📁 Results location:"
echo "   bench_runs/cert_v0.1_seed42/"
echo "   bench_runs/cert_v0.1_seed123/" 
echo "   bench_runs/cert_v0.1_seed456/"
echo ""
echo "🔍 Next steps:"
echo "   1. Analyze results:"
echo "      python scripts/analyze_safe_baselines.py --certifiable-v01 bench_runs/cert_v0.1_seed*"
echo ""
echo "   2. Validate policy compliance:"
echo "      ./scripts/pre_release_bench_check.sh"
echo ""
echo "📖 Documentation: gradience/bench/CERTIFIABLE_v0.1.md"