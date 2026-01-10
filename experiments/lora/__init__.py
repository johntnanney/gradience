"""
LoRA Experiments Suite

Validates our LoRA monitoring approach through systematic experiments:

1. exp1_rank_collapse: Demonstrate effective rank << nominal rank
2. exp2_optimal_rank: Find minimum rank for equivalent performance
3. exp3_alpha_sensitivity: Characterize α/r tradeoffs
4. exp4_task_complexity: How task difficulty affects required rank
5. exp5_layer_utilization: Which layers need more/less rank

Run all experiments:
    python experiments/lora/run_all.py
    
Quick mode (faster, fewer runs):
    python experiments/lora/run_all.py --quick

Run single experiment:
    python experiments/lora/run_all.py --experiment rank_collapse

Requirements:
    pip install transformers peft datasets evaluate accelerate
"""
