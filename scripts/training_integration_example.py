#!/usr/bin/env python3
"""
Example integration with popular training frameworks.

This shows how to adapt the validation protocol for different training setups:
- Hugging Face Transformers + PEFT
- Custom training loops
- Different model architectures

Modify the training commands in validation_protocol.py to use these patterns.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


class HuggingFaceTrainingConfig:
    """Generate Hugging Face Transformers + PEFT training configuration."""
    
    @staticmethod
    def create_config(
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        task_name: str,
        rank: int,
        rank_pattern: Dict[str, int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a complete training configuration for HF Transformers."""
        
        config = {
            # Model and task
            "model_name_or_path": model_name,
            "task_name": task_name,
            "dataset_name": dataset_name,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            
            # Training hyperparameters
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "per_device_train_batch_size": kwargs.get("batch_size", 8),
            "per_device_eval_batch_size": kwargs.get("eval_batch_size", 16),
            "num_train_epochs": kwargs.get("epochs", 3),
            "max_steps": kwargs.get("max_steps", -1),
            "warmup_steps": kwargs.get("warmup_steps", 0),
            "weight_decay": kwargs.get("weight_decay", 0.01),
            
            # Evaluation and logging
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": kwargs.get("logging_steps", 100),
            "load_best_model_at_end": True,
            "metric_for_best_model": kwargs.get("metric", "eval_accuracy"),
            "greater_is_better": True,
            
            # System settings
            "fp16": kwargs.get("use_fp16", True),
            "dataloader_num_workers": kwargs.get("num_workers", 4),
            "remove_unused_columns": False,
            "push_to_hub": False,
            
            # LoRA/PEFT configuration
            "use_peft": True,
            "peft_config": {
                "peft_type": "LORA",
                "task_type": "SEQ_CLS",  # or "CAUSAL_LM", "SEQ_2_SEQ_LM"
                "r": rank,
                "lora_alpha": rank * 2,  # Common 2x scaling
                "lora_dropout": 0.1,
                "target_modules": kwargs.get("target_modules", ["q_proj", "v_proj"]),
                "bias": "none",
                "fan_in_fan_out": False,
            }
        }
        
        # Add rank pattern for per-layer training
        if rank_pattern:
            config["peft_config"]["rank_pattern"] = rank_pattern
            
        return config
    
    @staticmethod
    def get_training_command(config_path: Path, script_path: str = "run_glue.py") -> List[str]:
        """Get command to run HF training script."""
        return [
            "python", script_path,
            f"--config_file={config_path}",
            "--do_train", "--do_eval"
        ]


class CustomTrainingConfig:
    """Configuration for custom training loops."""
    
    @staticmethod
    def create_pytorch_config(
        output_dir: Path,
        model_config: Dict[str, Any],
        rank: int,
        rank_pattern: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """Create configuration for PyTorch training loop."""
        
        config = {
            "model": model_config,
            "training": {
                "learning_rate": 3e-4,
                "batch_size": 8,
                "epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "gradient_clipping": 1.0,
                "scheduler": "cosine",
            },
            "lora": {
                "r": rank,
                "alpha": rank * 2,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "rank_pattern": rank_pattern
            },
            "output": {
                "output_dir": str(output_dir),
                "save_steps": 500,
                "eval_steps": 100,
                "log_steps": 50,
                "save_total_limit": 3
            },
            "system": {
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "num_workers": 4
            }
        }
        
        return config


class ValidationIntegration:
    """Helper methods for integrating validation with different training setups."""
    
    @staticmethod
    def adapt_for_transformers(base_config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Adapt configuration for specific Transformers model types."""
        
        config = base_config.copy()
        
        # Model-specific adaptations
        if "bert" in model_type.lower():
            config["target_modules"] = ["query", "value", "key", "dense"]
            config["task_type"] = "SEQ_CLS"
            
        elif "gpt" in model_type.lower():
            config["target_modules"] = ["c_attn", "c_proj"] 
            config["task_type"] = "CAUSAL_LM"
            
        elif "t5" in model_type.lower():
            config["target_modules"] = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
            config["task_type"] = "SEQ_2_SEQ_LM"
            
        elif "distilbert" in model_type.lower():
            config["target_modules"] = ["q_lin", "v_lin", "k_lin", "out_lin"]
            config["task_type"] = "SEQ_CLS"
        
        return config
    
    @staticmethod
    def create_rank_pattern_mapping(
        suggestions: Dict[str, Any], 
        model_architecture: str
    ) -> Dict[str, int]:
        """Convert gradience suggestions to framework-specific rank patterns."""
        
        rank_pattern = suggestions.get("rank_pattern", {})
        if not rank_pattern:
            return {}
        
        # Map gradience layer names to framework layer names
        framework_pattern = {}
        
        for gradience_name, rank in rank_pattern.items():
            # Convert from gradience format to framework format
            # Example: "base_model.model.bert.encoder.layer.0.attention.self.query"
            # to format expected by PEFT
            
            if model_architecture == "bert":
                # PEFT expects: "bert.encoder.layer.0.attention.self.query"
                if "base_model.model." in gradience_name:
                    framework_name = gradience_name.replace("base_model.model.", "")
                    framework_pattern[framework_name] = rank
                    
            elif model_architecture == "distilbert":
                # PEFT expects: "distilbert.transformer.layer.0.attention.q_lin"
                if "base_model.model." in gradience_name:
                    framework_name = gradience_name.replace("base_model.model.", "")
                    framework_pattern[framework_name] = rank
                    
        return framework_pattern
    
    @staticmethod
    def estimate_memory_usage(
        base_params: int,
        rank: int,
        num_layers: int,
        modules_per_layer: int = 4
    ) -> Dict[str, int]:
        """Estimate memory usage for different rank configurations."""
        
        # LoRA parameters: A (rank × in_dim) + B (out_dim × rank)
        # Assuming square attention matrices for simplicity
        hidden_dim = 768  # Common size
        
        lora_params = num_layers * modules_per_layer * 2 * rank * hidden_dim
        total_params = base_params + lora_params
        
        # Rough memory estimates (in MB)
        param_memory = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        activation_memory = param_memory * 2  # Rough estimate
        optimizer_memory = param_memory * 2  # Adam stores momentum
        
        return {
            "lora_params": lora_params,
            "total_params": total_params, 
            "param_memory_mb": int(param_memory),
            "total_memory_mb": int(param_memory + activation_memory + optimizer_memory)
        }


def example_usage():
    """Example of how to use these configurations."""
    
    # Example 1: Create HF Transformers config
    hf_config = HuggingFaceTrainingConfig.create_config(
        output_dir=Path("./output/experiment_1"),
        model_name="distilbert-base-uncased",
        dataset_name="glue",
        task_name="cola",
        rank=8,
        learning_rate=3e-4,
        epochs=3,
        batch_size=16
    )
    
    print("HF Transformers config:")
    print(json.dumps(hf_config, indent=2))
    
    # Example 2: Convert gradience suggestions to PEFT rank pattern
    mock_suggestions = {
        "default_r": 4,
        "rank_pattern": {
            "base_model.model.distilbert.transformer.layer.0.attention.q_lin": 2,
            "base_model.model.distilbert.transformer.layer.0.attention.k_lin": 8,
            "base_model.model.distilbert.transformer.layer.1.attention.q_lin": 2,
        }
    }
    
    rank_pattern = ValidationIntegration.create_rank_pattern_mapping(
        mock_suggestions, "distilbert"
    )
    
    print("\nConverted rank pattern:")
    print(json.dumps(rank_pattern, indent=2))
    
    # Example 3: Memory estimation
    memory_est = ValidationIntegration.estimate_memory_usage(
        base_params=66_000_000,  # DistilBERT base
        rank=8,
        num_layers=6,
        modules_per_layer=4
    )
    
    print("\nMemory estimation:")
    print(json.dumps(memory_est, indent=2))


if __name__ == "__main__":
    example_usage()