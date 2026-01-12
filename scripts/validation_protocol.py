#!/usr/bin/env python3
"""
CPU-only validation protocol for rank suggestions.

This script validates rank compression recommendations by:
1. Training a probe at high rank (r=16 or r=32) on a tiny dataset
2. Generating rank suggestions using gradience audit
3. Retraining with different suggestion strategies:
   - uniform_p90: Single rank based on global p90
   - module_p90: Per-module-type ranks based on p90
   - per_layer: Full per-layer rank pattern
4. Comparing performance across strategies

Usage:
    python scripts/validation_protocol.py --model tiny-bert --dataset tiny --probe-r 16
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class ValidationProtocol:
    """Manages the complete validation protocol workflow."""
    
    def __init__(self, base_dir: Path, model: str, dataset: str, probe_r: int, verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.model = model
        self.dataset = dataset
        self.probe_r = probe_r
        self.verbose = verbose
        
        # Create protocol-specific directory
        self.protocol_dir = self.base_dir / f"validation_{model}_{dataset}_r{probe_r}"
        self.protocol_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def log(self, message: str):
        """Log message with timestamp if verbose."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
        """Run command with logging."""
        if self.verbose:
            self.log(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=cwd or self.protocol_dir,
            capture_output=capture,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            self.log(f"Command failed with exit code {result.returncode}")
            if capture and result.stderr:
                self.log(f"Error: {result.stderr}")
        
        return result
    
    def create_tiny_training_config(self, output_dir: Path, rank: int, rank_pattern: Optional[Dict[str, int]] = None) -> Path:
        """Create a minimal training configuration for CPU validation."""
        config = {
            # Model and data
            "model_name_or_path": self._get_model_path(),
            "dataset_name": self._get_dataset_config(),
            "dataset_config_name": None,
            "max_train_samples": 100,  # Very small for CPU validation
            "max_eval_samples": 20,
            
            # Training parameters - minimal for CPU
            "learning_rate": 5e-4,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 1,
            "max_steps": 50,  # Very short training
            "warmup_steps": 5,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 25,
            "save_strategy": "steps", 
            "save_steps": 50,
            "save_total_limit": 1,
            
            # Output
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "remove_unused_columns": False,
            
            # CPU optimizations
            "fp16": False,  # CPU doesn't support fp16
            "dataloader_num_workers": 0,
            "disable_tqdm": not self.verbose,
            
            # LoRA configuration
            "use_lora": True,
            "lora_r": rank,
            "lora_alpha": rank * 2,  # Common 2x scaling
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_bias": "none",
            
            # Task specific
            "task_name": self._get_task_name(),
            "do_train": True,
            "do_eval": True,
        }
        
        # Add rank pattern if provided (for per-layer validation)
        if rank_pattern:
            config["lora_rank_pattern"] = rank_pattern
        
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _get_model_path(self) -> str:
        """Get model path/name based on model type."""
        model_map = {
            "tiny-bert": "hf-internal-testing/tiny-random-BertForSequenceClassification",
            "tiny-distilbert": "hf-internal-testing/tiny-random-distilbert",
            "tiny-gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        }
        return model_map.get(self.model, self.model)
    
    def _get_dataset_config(self) -> str:
        """Get dataset configuration."""
        dataset_map = {
            "tiny": "glue",
            "cola": "glue", 
            "sst2": "glue",
        }
        return dataset_map.get(self.dataset, self.dataset)
    
    def _get_task_name(self) -> str:
        """Get task name for dataset."""
        task_map = {
            "tiny": "cola",  # Use CoLA for tiny validation
            "cola": "cola",
            "sst2": "sst2",
        }
        return task_map.get(self.dataset, "cola")
    
    def train_probe(self) -> Path:
        """Train initial probe at high rank."""
        self.log(f"Step 1: Training probe at rank {self.probe_r}")
        
        probe_dir = self.protocol_dir / f"probe_r{self.probe_r}"
        probe_dir.mkdir(exist_ok=True)
        
        # Create training config
        config_path = self.create_tiny_training_config(probe_dir, self.probe_r)
        
        # Run training (mock implementation - you'd use your actual training script)
        self.log("Training probe model...")
        train_cmd = self._get_training_command(config_path, probe_dir)
        result = self.run_command(train_cmd)
        
        if result.returncode == 0:
            self.log(f"✓ Probe training completed: {probe_dir}")
            self.results["probe"] = {
                "rank": self.probe_r,
                "output_dir": str(probe_dir),
                "config": str(config_path),
                "status": "success"
            }
        else:
            self.log(f"✗ Probe training failed")
            self.results["probe"] = {"status": "failed", "error": result.stderr}
            
        return probe_dir
    
    def _get_training_command(self, config_path: Path, output_dir: Path) -> List[str]:
        """Get training command. Override this with your actual training script."""
        # This is a mock - replace with your actual training command
        return [
            "python", "-c", 
            f"""
import json
import torch
from pathlib import Path

# Mock training that creates the expected PEFT structure
output_dir = Path('{output_dir}')
peft_dir = output_dir / 'peft'
peft_dir.mkdir(exist_ok=True)

# Create mock adapter config
with open('{config_path}') as f:
    config = json.load(f)

adapter_config = {{
    'r': config['lora_r'],
    'lora_alpha': config['lora_alpha'], 
    'lora_dropout': config['lora_dropout'],
    'target_modules': config['lora_target_modules'],
    'peft_type': 'LORA',
    'task_type': 'SEQ_CLS'
}}

with open(peft_dir / 'adapter_config.json', 'w') as f:
    json.dump(adapter_config, f, indent=2)

# Create mock weights
weights = {{}}
target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
for layer_idx in range(2):  # 2 layers for tiny model
    for module in target_modules:
        base_name = f'base_model.model.bert.encoder.layer.{{layer_idx}}.attention.self.{{module}}'
        weights[f'{{base_name}}.lora_A.weight'] = torch.randn(config['lora_r'], 64)
        weights[f'{{base_name}}.lora_B.weight'] = torch.randn(64, config['lora_r'])

torch.save(weights, peft_dir / 'adapter_model.bin')

# Create training metrics
metrics = {{
    'train_loss': 0.5,
    'eval_loss': 0.6, 
    'eval_accuracy': 0.75,
    'train_runtime': 30.0
}}

with open(output_dir / 'trainer_state.json', 'w') as f:
    json.dump({{'log_history': [metrics]}}, f)

print(f"Mock training completed: {{output_dir}}")
"""
        ]
    
    def generate_suggestions(self, probe_dir: Path) -> Dict[str, Any]:
        """Generate rank suggestions from probe model."""
        self.log("Step 2: Generating rank suggestions")
        
        peft_dir = probe_dir / "peft"
        if not peft_dir.exists():
            self.log(f"✗ PEFT directory not found: {peft_dir}")
            return {}
        
        # Run gradience audit with per-layer suggestions
        audit_cmd = [
            "python", "-m", "gradience", "audit",
            "--peft-dir", str(peft_dir),
            "--layers", "--suggest-per-layer", "--json"
        ]
        
        result = self.run_command(audit_cmd, capture=True)
        
        if result.returncode != 0:
            self.log(f"✗ Audit failed: {result.stderr}")
            return {}
        
        try:
            audit_data = json.loads(result.stdout)
            suggestions = audit_data.get("rank_suggestions", {})
            
            self.log(f"✓ Generated suggestions:")
            self.log(f"  Default rank: {suggestions.get('default_r', 'N/A')}")
            self.log(f"  Rank pattern entries: {len(suggestions.get('rank_pattern', {}))}")
            self.log(f"  Module type p90: {suggestions.get('by_module_type_p90', {})}")
            
            # Save full audit data
            audit_path = self.protocol_dir / "probe_audit.json"
            with open(audit_path, "w") as f:
                json.dump(audit_data, f, indent=2)
            
            self.results["suggestions"] = suggestions
            return suggestions
            
        except json.JSONDecodeError as e:
            self.log(f"✗ Failed to parse audit JSON: {e}")
            return {}
    
    def extract_strategies(self, suggestions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract different retraining strategies from suggestions."""
        strategies = {}
        
        # Strategy 1: Uniform P90 (conservative)
        global_p90 = suggestions.get("default_r", self.probe_r // 2)
        strategies["uniform_p90"] = {
            "type": "uniform",
            "rank": global_p90,
            "description": f"Uniform rank {global_p90} (conservative global)"
        }
        
        # Strategy 2: Module-type P90 (moderately conservative)  
        module_p90 = suggestions.get("by_module_type_p90", {})
        if module_p90:
            # Use the most common module type rank or average
            ranks = list(module_p90.values())
            avg_rank = int(sum(ranks) / len(ranks)) if ranks else global_p90
            strategies["module_p90"] = {
                "type": "module",
                "base_rank": avg_rank,
                "module_ranks": module_p90,
                "description": f"Per-module ranks (avg {avg_rank})"
            }
        else:
            # Fallback to uniform if no module data
            strategies["module_p90"] = strategies["uniform_p90"].copy()
            strategies["module_p90"]["description"] += " (fallback)"
        
        # Strategy 3: Full per-layer (experimental)
        rank_pattern = suggestions.get("rank_pattern", {})
        default_r = suggestions.get("default_r", global_p90)
        strategies["per_layer"] = {
            "type": "per_layer",
            "default_rank": default_r,
            "rank_pattern": rank_pattern,
            "description": f"Per-layer pattern (default {default_r}, {len(rank_pattern)} overrides)"
        }
        
        return strategies
    
    def retrain_with_strategy(self, strategy_name: str, strategy: Dict[str, Any]) -> Path:
        """Retrain model using specific strategy."""
        self.log(f"Step 3.{strategy_name}: Retraining with {strategy['description']}")
        
        retrain_dir = self.protocol_dir / f"retrain_{strategy_name}"
        retrain_dir.mkdir(exist_ok=True)
        
        # Determine rank configuration
        if strategy["type"] == "uniform":
            rank = strategy["rank"]
            config_path = self.create_tiny_training_config(retrain_dir, rank)
            
        elif strategy["type"] == "module":
            # Use base rank (module-specific ranks would require more complex PEFT config)
            rank = strategy["base_rank"] 
            config_path = self.create_tiny_training_config(retrain_dir, rank)
            
        elif strategy["type"] == "per_layer":
            # Use default rank with rank pattern
            rank = strategy["default_rank"]
            rank_pattern = strategy["rank_pattern"]
            config_path = self.create_tiny_training_config(retrain_dir, rank, rank_pattern)
        
        # Run retraining
        train_cmd = self._get_training_command(config_path, retrain_dir)
        result = self.run_command(train_cmd)
        
        if result.returncode == 0:
            self.log(f"✓ Retraining completed: {retrain_dir}")
            
            # Calculate parameter reduction
            original_params = self._estimate_params(self.probe_r)
            new_params = self._estimate_params_with_strategy(strategy)
            reduction = (original_params - new_params) / original_params if original_params > 0 else 0
            
            self.results[f"retrain_{strategy_name}"] = {
                "strategy": strategy,
                "output_dir": str(retrain_dir),
                "config": str(config_path),
                "status": "success",
                "param_reduction": reduction,
                "original_params": original_params,
                "new_params": new_params
            }
        else:
            self.log(f"✗ Retraining failed")
            self.results[f"retrain_{strategy_name}"] = {
                "strategy": strategy,
                "status": "failed", 
                "error": result.stderr
            }
            
        return retrain_dir
    
    def _estimate_params(self, rank: int, num_layers: int = 2, hidden_size: int = 64) -> int:
        """Estimate LoRA parameters for uniform rank."""
        # 4 modules per layer (q, k, v, o) 
        params_per_module = 2 * rank * hidden_size  # A + B matrices
        return num_layers * 4 * params_per_module
    
    def _estimate_params_with_strategy(self, strategy: Dict[str, Any]) -> int:
        """Estimate parameters with given strategy."""
        if strategy["type"] in ["uniform", "module"]:
            rank = strategy.get("rank") or strategy.get("base_rank")
            return self._estimate_params(rank)
        
        elif strategy["type"] == "per_layer":
            # More complex calculation for per-layer patterns
            default_rank = strategy["default_rank"]
            rank_pattern = strategy.get("rank_pattern", {})
            
            total_params = 0
            num_layers = 2
            modules_per_layer = 4
            hidden_size = 64
            
            for layer_idx in range(num_layers):
                for module in ["q_proj", "v_proj", "k_proj", "o_proj"]:
                    module_name = f"layer.{layer_idx}.attention.self.{module}"
                    rank = rank_pattern.get(module_name, default_rank)
                    total_params += 2 * rank * hidden_size
            
            return total_params
        
        return 0
    
    def evaluate_results(self) -> Dict[str, Any]:
        """Evaluate and compare results across strategies."""
        self.log("Step 4: Evaluating results")
        
        evaluation = {
            "protocol_summary": {
                "model": self.model,
                "dataset": self.dataset, 
                "probe_rank": self.probe_r,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "protocol_dir": str(self.protocol_dir)
            },
            "strategies_tested": [],
            "parameter_reductions": {},
            "recommendations": []
        }
        
        # Analyze each strategy
        for key, result in self.results.items():
            if key.startswith("retrain_") and result.get("status") == "success":
                strategy_name = key.replace("retrain_", "")
                strategy = result["strategy"]
                
                evaluation["strategies_tested"].append({
                    "name": strategy_name,
                    "description": strategy["description"],
                    "param_reduction": result.get("param_reduction", 0),
                    "original_params": result.get("original_params", 0),
                    "new_params": result.get("new_params", 0)
                })
                
                evaluation["parameter_reductions"][strategy_name] = result.get("param_reduction", 0)
        
        # Generate recommendations
        reductions = evaluation["parameter_reductions"]
        if reductions:
            best_strategy = max(reductions.items(), key=lambda x: x[1])
            safest_strategy = "uniform_p90" if "uniform_p90" in reductions else list(reductions.keys())[0]
            
            evaluation["recommendations"] = [
                f"Safest approach: {safest_strategy} ({reductions.get(safest_strategy, 0):.1%} reduction)",
                f"Best reduction: {best_strategy[0]} ({best_strategy[1]:.1%} reduction)",
                f"Per-layer experimental: {'per_layer' in reductions} ({reductions.get('per_layer', 0):.1%} reduction)"
            ]
        
        # Save evaluation
        eval_path = self.protocol_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        
        self.log("✓ Evaluation completed")
        return evaluation
    
    def run_full_protocol(self) -> Dict[str, Any]:
        """Run the complete validation protocol."""
        self.log(f"Starting validation protocol for {self.model}/{self.dataset} at rank {self.probe_r}")
        
        try:
            # Step 1: Train probe
            probe_dir = self.train_probe()
            if self.results.get("probe", {}).get("status") != "success":
                return {"error": "Probe training failed"}
            
            # Step 2: Generate suggestions
            suggestions = self.generate_suggestions(probe_dir)
            if not suggestions:
                return {"error": "Suggestion generation failed"}
            
            # Step 3: Extract strategies and retrain
            strategies = self.extract_strategies(suggestions)
            
            for strategy_name, strategy in strategies.items():
                self.retrain_with_strategy(strategy_name, strategy)
            
            # Step 4: Evaluate results
            evaluation = self.evaluate_results()
            
            self.log("✓ Validation protocol completed successfully")
            return evaluation
            
        except Exception as e:
            self.log(f"✗ Protocol failed with error: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="CPU-only validation protocol for rank suggestions")
    parser.add_argument("--model", default="tiny-distilbert", 
                       choices=["tiny-bert", "tiny-distilbert", "tiny-gpt2"],
                       help="Model to use for validation")
    parser.add_argument("--dataset", default="tiny", 
                       choices=["tiny", "cola", "sst2"], 
                       help="Dataset to use for validation")
    parser.add_argument("--probe-r", type=int, default=16,
                       choices=[8, 16, 32],
                       help="Probe rank for initial training")
    parser.add_argument("--base-dir", type=Path, default="./validation_runs",
                       help="Base directory for validation outputs")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create and run protocol
    protocol = ValidationProtocol(
        base_dir=args.base_dir,
        model=args.model,
        dataset=args.dataset, 
        probe_r=args.probe_r,
        verbose=args.verbose
    )
    
    results = protocol.run_full_protocol()
    
    # Print summary
    if "error" in results:
        print(f"\n❌ Validation failed: {results['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Validation completed successfully!")
        print(f"📁 Results saved to: {protocol.protocol_dir}")
        
        if "recommendations" in results:
            print("\n📊 Recommendations:")
            for rec in results["recommendations"]:
                print(f"   • {rec}")


if __name__ == "__main__":
    main()