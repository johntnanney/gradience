#!/usr/bin/env python3
"""
Create a test PEFT adapter directory for testing Step 4 integration.
"""

import torch
import json
from pathlib import Path

def create_test_peft_adapter(output_dir: str):
    """Create a realistic test PEFT adapter directory."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create adapter_config.json
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16.0,
        "target_modules": [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create adapter weights with realistic structure
    state_dict = {}
    
    # Attention layers with different characteristics
    layers = [
        ("model.layers.0.self_attn.q_proj", [10.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1]),  # Clear low rank
        ("model.layers.0.self_attn.k_proj", [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5]),  # Gradual decay
        ("model.layers.0.self_attn.v_proj", [15.0, 8.0, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]),  # High noise
        ("model.layers.0.self_attn.o_proj", [6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.06]),  # Exponential decay
        ("model.layers.1.mlp.gate_proj", [12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5]),  # Rich structure
        ("model.layers.1.mlp.up_proj", [20.0, 2.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05]),  # One dominant
        ("model.layers.1.mlp.down_proj", [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5]),  # Almost uniform
    ]
    
    for module_name, singular_values in layers:
        r = len(singular_values)
        
        # Create A and B matrices with controlled singular values
        A = torch.randn(r, 4096)  # Typical input dimension
        U, _, Vt = torch.linalg.svd(A, full_matrices=False)
        s = torch.tensor(singular_values, dtype=torch.float32)
        A = (U * s.unsqueeze(0)) @ Vt
        
        B = torch.randn(4096, r) * 0.1  # Typical output dimension
        
        state_dict[f"{module_name}.lora_A.weight"] = A
        state_dict[f"{module_name}.lora_B.weight"] = B
    
    # Save as safetensors (preferred) or pytorch format
    try:
        import safetensors.torch
        safetensors.torch.save_file(state_dict, output_path / "adapter_model.safetensors")
        print(f"Created adapter with safetensors format")
    except ImportError:
        torch.save(state_dict, output_path / "adapter_model.bin")
        print(f"Created adapter with pytorch format")
    
    print(f"Created test PEFT adapter at: {output_path}")
    print(f"  - Config: {len(config)} parameters")
    print(f"  - Weights: {len(state_dict)//2} LoRA layers")
    print(f"  - Total parameters: {sum(t.numel() for t in state_dict.values()):,}")
    
    return output_path

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_adapter"
    create_test_peft_adapter(output_dir)