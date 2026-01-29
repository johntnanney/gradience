#!/usr/bin/env python3
"""
Demo: Policy Scoreboard - Self-Improving Framework

Shows how the policy scoreboard transforms Gradience from "policies as hypotheses" 
into "policies with track records" by tracking performance across benchmark runs.
"""

import sys
sys.path.insert(0, '.')

import tempfile
from pathlib import Path
from gradience.vnext.policy_scoreboard import PolicyScoreboard, PolicyBenchmarkResult
from datetime import datetime


def create_demo_benchmark_results():
    """Create realistic demo benchmark results for multiple policies."""
    
    # Simulate 15 benchmark runs across different models/tasks
    results = []
    
    benchmark_scenarios = [
        ("distilbert_sst2", "distilbert-base-uncased", "sst2"),
        ("distilbert_sst2", "distilbert-base-uncased", "sst2"),  # Multiple seeds
        ("distilbert_sst2", "distilbert-base-uncased", "sst2"),
        ("roberta_imdb", "roberta-base", "imdb"),
        ("roberta_imdb", "roberta-base", "imdb"),
        ("roberta_imdb", "roberta-base", "imdb"),
        ("bert_cola", "bert-base-uncased", "cola"),
        ("bert_cola", "bert-base-uncased", "cola"),
        ("gpt2_wikitext", "gpt2", "wikitext"),
        ("gpt2_wikitext", "gpt2", "wikitext"),
        ("t5_summarization", "t5-small", "xsum"),
        ("t5_summarization", "t5-small", "xsum"),
        ("llama_qa", "llama2-7b", "squad"),
        ("llama_qa", "llama2-7b", "squad"),
        ("llama_qa", "llama2-7b", "squad"),
    ]
    
    for i, (config, model, task) in enumerate(benchmark_scenarios):
        # Simulate different policy performances with realistic characteristics
        
        # Energy policy: Reliable, slightly conservative, high performance
        energy_optimal = 6 if "bert" in model else 8
        energy_result = PolicyBenchmarkResult(
            config_name=config,
            date=datetime.now().isoformat(),
            model_name=model,
            task_name=task,
            policy_name="energy_90",
            suggested_rank=8,
            actual_rank=8,
            optimal_rank=energy_optimal,
            passed=True if i % 5 != 0 else False,  # 80% pass rate
            performance_delta=0.024 + (i % 3) * 0.005,  # 2.4% - 3.4% performance
            best_candidate_delta=0.028,  # Best among all candidates
            seed=42 + i
        )
        results.append(energy_result)
        
        # Knee policy: More aggressive, finds elbows early, lower reliability  
        knee_optimal = 4 if "gpt" in model else 5
        knee_result = PolicyBenchmarkResult(
            config_name=config,
            date=datetime.now().isoformat(),
            model_name=model,
            task_name=task,
            policy_name="knee",
            suggested_rank=4,
            actual_rank=4,
            optimal_rank=knee_optimal,
            passed=True if i % 3 != 0 else False,  # 67% pass rate
            performance_delta=0.018 + (i % 4) * 0.008,  # More variable: 1.8% - 4.2%
            best_candidate_delta=0.028,
            seed=42 + i
        )
        results.append(knee_result)
        
        # eRank policy: Good balance, improving over time
        erank_optimal = 6 if "t5" in model else 7
        base_perf = 0.020 + min(i * 0.001, 0.008)  # Improving trend
        erank_result = PolicyBenchmarkResult(
            config_name=config,
            date=datetime.now().isoformat(),
            model_name=model,
            task_name=task,
            policy_name="erank",
            suggested_rank=6,
            actual_rank=6,
            optimal_rank=erank_optimal,
            passed=True if i % 4 != 0 else False,  # 75% pass rate
            performance_delta=base_perf + (i % 2) * 0.003,
            best_candidate_delta=0.028,
            seed=42 + i
        )
        results.append(erank_result)
        
        # OHT policy: Very conservative, high reliability, sometimes too safe
        oht_optimal = 8 if "llama" in model else 10
        oht_result = PolicyBenchmarkResult(
            config_name=config,
            date=datetime.now().isoformat(),
            model_name=model,
            task_name=task,
            policy_name="oht",
            suggested_rank=10,
            actual_rank=10,
            optimal_rank=oht_optimal,
            passed=True if i % 8 != 0 else False,  # 87% pass rate
            performance_delta=0.022 + (i % 2) * 0.002,  # Consistent performance
            best_candidate_delta=0.028,
            seed=42 + i
        )
        results.append(oht_result)
    
    return results


def demonstrate_policy_scoreboard():
    """Demonstrate the complete policy scoreboard functionality."""
    
    print("🧠 Policy Scoreboard Demo - Self-Improving Framework")
    print("=" * 65)
    print("Transform 'policies as hypotheses' → 'policies with track records'")
    print()
    
    # Create temporary scoreboard for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        scoreboard_path = Path(temp_dir) / "demo_scoreboard.json"
        scoreboard = PolicyScoreboard(scoreboard_path)
        
        print("📊 Simulating 15 benchmark runs across different models/tasks...")
        
        # Add benchmark results in groups (simulating multiple benchmark runs)
        demo_results = create_demo_benchmark_results()
        
        # Group results by benchmark run (4 policies per run)
        for i in range(0, len(demo_results), 4):
            batch = demo_results[i:i+4]
            if len(batch) >= 4:
                config = batch[0].config_name
                model = batch[0].model_name
                task = batch[0].task_name
                
                print(f"   Benchmark #{i//4 + 1}: {config} ({model} on {task})")
                scoreboard.add_benchmark_results(config, model, task, batch)
        
        print(f"\n✅ Processed {len(demo_results)} policy results across {len(demo_results)//4} benchmarks")
        print()
        
        # Show the generated scoreboard data structure
        print("💾 GENERATED SCOREBOARD JSON STRUCTURE:")
        print("=" * 50)
        print(f"📁 Stored at: {scoreboard.scoreboard_path}")
        print(f"📈 Total benchmarks: {scoreboard.data['total_benchmarks']}")
        print(f"🎯 Policies tracked: {len(scoreboard.data['policies'])}")
        print()
        
        # Show key metrics for each policy
        print("📊 POLICY PERFORMANCE SUMMARY:")
        print("=" * 50)
        
        best_policies = scoreboard.get_best_policies(10)
        for i, (policy_name, metrics) in enumerate(best_policies):
            print(f"{i+1}. {policy_name}:")
            print(f"   Pass Rate: {metrics.pass_rate:.1%} ({metrics.passes}/{metrics.total_attempts})")
            print(f"   Optimality: {metrics.avg_optimality:.1%}")
            print(f"   Bias: {metrics.conservatism_bias:+.1%} ({'Conservative' if metrics.conservatism_bias > 0 else 'Aggressive'})")
            print(f"   Reliability: {metrics.reliability_score:.1%}")
            print(f"   Trend: {metrics.trend}")
            print(f"   Note: {metrics.note}")
            print()
        
        # Show summary insights  
        insights = scoreboard.generate_summary_insights()
        print("🎯 FRAMEWORK INSIGHTS:")
        print("=" * 50)
        print(f"Best Overall: {insights.get('best_overall_policy', 'N/A')}")
        print(f"Most Reliable: {insights.get('most_reliable', 'N/A')}")
        print(f"Most Aggressive: {insights.get('most_aggressive', 'N/A')}")
        print(f"Most Conservative: {insights.get('most_conservative', 'N/A')}")
        print()
        print("Recommendations:")
        for rec in insights.get("recommendations", []):
            print(f"  • {rec}")
        print()
        
        # Generate and show the markdown table
        print("📝 MARKDOWN TABLE FOR REPORTS:")
        print("=" * 50)
        markdown_table = scoreboard.generate_markdown_table()
        print(markdown_table)
        print()
        
        # Show how this transforms the framework
        print("🚀 TRANSFORMATION: POLICIES AS HYPOTHESES → POLICIES WITH TRACK RECORDS")
        print("=" * 80)
        
        print("\n❌ BEFORE (Static Policy System):")
        print("   • 'Try energy_90 policy... maybe it works?'")
        print("   • 'Knee policy seems aggressive, but how aggressive?'")
        print("   • 'No data on which policy works best for my model type'")
        print("   • 'Can't tell if policies are improving or declining'")
        
        print("\n✅ AFTER (Self-Improving Framework):")
        print("   • 'energy_90 has 80% pass rate, 91% optimality - proven reliable'")
        print("   • 'knee is -15% biased (aggressive) but good for max compression'")
        print("   • 'For transformer models, energy_90 consistently outperforms'")  
        print("   • 'erank shows improving trend - getting better with more data'")
        
        print("\n🎯 KILLER BENEFITS:")
        print("   1. EVIDENCE-BASED SELECTION: Choose policies with proven track records")
        print("   2. CONTINUOUS IMPROVEMENT: Policies get better as more data collected")
        print("   3. BIAS AWARENESS: Understand when policies are too aggressive/conservative")
        print("   4. MODEL-SPECIFIC INSIGHTS: See which policies work best for different models")
        print("   5. RESEARCH ACCELERATION: Quickly identify promising policy directions")
        
        print("\n💡 USAGE SCENARIOS:")
        print()
        print("🏭 PRACTITIONER:")
        print('   > "I need reliable compression for production."')
        print('   > Scoreboard shows energy_90 has 91% optimality, 88% reliability')
        print('   > Decision: Use energy_90 ✅')
        
        print("\n🔬 RESEARCHER:")  
        print('   > "I\'m exploring aggressive compression."')
        print('   > Scoreboard shows knee has -15% bias but only 67% pass rate')
        print('   > Decision: Investigate knee failure modes 🔍')
        
        print("\n🛠️  FRAMEWORK DEVELOPER:")
        print('   > "Are current policies effective on new model types?"')
        print('   > Scoreboard shows all policies trending down on transformers')
        print('   > Decision: Develop transformer-specific policies 🚧')
        
        print("\n" + "=" * 80)
        print("🎉 POLICY SCOREBOARD: FRAMEWORK BECOMES SELF-IMPROVING!")
        print("Transform from static algorithms → learning system with evidence")


if __name__ == "__main__":
    demonstrate_policy_scoreboard()