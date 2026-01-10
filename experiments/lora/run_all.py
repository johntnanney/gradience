"""
LoRA Experiments Suite

Validates our LoRA monitoring approach and generates calibration data.

Experiments:
1. Rank Collapse Demo - Show effective rank << nominal rank
2. Optimal Rank Search - Find minimum rank for equivalent performance  
3. Alpha Sensitivity - Characterize α/r tradeoffs
4. Task Complexity vs Rank - How does task affect required rank?
5. Layer-wise Utilization - Which layers need more/less rank?

These run on modest hardware (single GPU, 8-16GB VRAM).
Target models: DistilBERT, BERT-base, RoBERTa-base

Usage:
    # Run all experiments
    python experiments/lora/run_all.py
    
    # Run specific experiment
    python experiments/lora/run_all.py --experiment rank_collapse
    
    # Quick mode (fewer runs)
    python experiments/lora/run_all.py --quick

Requirements:
    pip install transformers datasets evaluate peft accelerate
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime


def run_rank_collapse_experiment(output_dir: Path, quick: bool = False):
    """Experiment 1: Demonstrate rank collapse."""
    from experiments.lora.exp1_rank_collapse import run_experiment
    return run_experiment(output_dir, quick=quick)


def run_optimal_rank_experiment(output_dir: Path, quick: bool = False):
    """Experiment 2: Find optimal rank."""
    from experiments.lora.exp2_optimal_rank import run_experiment
    return run_experiment(output_dir, quick=quick)


def run_alpha_sensitivity_experiment(output_dir: Path, quick: bool = False):
    """Experiment 3: Alpha/rank sensitivity."""
    from experiments.lora.exp3_alpha_sensitivity import run_experiment
    return run_experiment(output_dir, quick=quick)


def run_task_complexity_experiment(output_dir: Path, quick: bool = False):
    """Experiment 4: Task complexity vs required rank."""
    from experiments.lora.exp4_task_complexity import run_experiment
    return run_experiment(output_dir, quick=quick)


def run_layer_utilization_experiment(output_dir: Path, quick: bool = False):
    """Experiment 5: Layer-wise utilization patterns."""
    from experiments.lora.exp5_layer_utilization import run_experiment
    return run_experiment(output_dir, quick=quick)


EXPERIMENTS = {
    'rank_collapse': run_rank_collapse_experiment,
    'optimal_rank': run_optimal_rank_experiment,
    'alpha_sensitivity': run_alpha_sensitivity_experiment,
    'task_complexity': run_task_complexity_experiment,
    'layer_utilization': run_layer_utilization_experiment,
}


def main():
    parser = argparse.ArgumentParser(description="LoRA Experiments Suite")
    parser.add_argument(
        '--experiment', 
        type=str, 
        choices=list(EXPERIMENTS.keys()) + ['all'],
        default='all',
        help='Which experiment to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./lora_experiments',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with fewer runs'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log run info
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'experiment': args.experiment,
        'quick_mode': args.quick,
    }
    
    print("=" * 60)
    print("LORA EXPERIMENTS SUITE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print()
    
    results = {}
    
    if args.experiment == 'all':
        experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = {args.experiment: EXPERIMENTS[args.experiment]}
    
    for name, run_fn in experiments_to_run.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}\n")
        
        exp_output = output_dir / name
        exp_output.mkdir(exist_ok=True)
        
        try:
            result = run_fn(exp_output, quick=args.quick)
            results[name] = {'status': 'success', 'result': result}
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            results[name] = {'status': 'error', 'error': str(e)}
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    run_info['results'] = results
    with open(output_dir / 'run_summary.json', 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status} {name}: {result['status']}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
