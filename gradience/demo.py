"""
Gradience Demo: 1-Command Validation

Run with:
    python -m gradience.demo --guard --inject-nan

Demonstrates:
- Spectral monitoring (κ̃ tracking)
- Guard system (checkpoint/rollback)
- Automatic recovery from NaN injection
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import time


def check_deps() -> bool:
    """Check required dependencies."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        print("   Install: pip install torch transformers")
        return False
    return True


def create_dataset(tokenizer, n_samples=100, seq_len=64):
    """Create synthetic dataset."""
    import torch
    from torch.utils.data import Dataset
    
    class SyntheticDataset(Dataset):
        def __init__(self, tokenizer, n, seq_len):
            self.n = n
            self.seq_len = seq_len
            self.vocab_size = tokenizer.vocab_size
            self.data = torch.randint(0, self.vocab_size, (n, seq_len))
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            ids = self.data[idx]
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": ids.clone(),
            }
    
    return SyntheticDataset(tokenizer, n_samples, seq_len)


def nan_injector_callback(inject_step: int):
    """Create callback that injects NaN at specific step."""
    from transformers import TrainerCallback
    import torch
    
    class NaNInjector(TrainerCallback):
        def __init__(self, step):
            self.step = step
            self.done = False
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step == self.step and not self.done:
                print(f"\n🔥 INJECTING NaN at step {self.step}...")
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        with torch.no_grad():
                            p.data[0].fill_(float('nan'))
                        print(f"   Corrupted: {name}")
                        break
                self.done = True
    
    return NaNInjector(inject_step)


def run_demo(
    guard: bool = True,
    inject_nan: bool = False,
    inject_step: int = 50,
    max_steps: int = 100,
    out_dir: str = "./gradience_demo",
):
    """Run demo training."""
    import torch
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        Trainer,
        TrainingArguments,
    )
    from gradience import GradienceCallback
    
    print("=" * 60)
    print("GRADIENCE DEMO")
    print("=" * 60)
    print(f"  Guard:       {guard}")
    print(f"  Inject NaN:  {inject_nan}" + (f" @ step {inject_step}" if inject_nan else ""))
    print(f"  Max steps:   {max_steps}")
    print(f"  Output:      {out_dir}")
    print("=" * 60)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Tiny model
    print("\n📦 Loading tiny GPT-2...")
    config = GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {n_params:,}")
    
    # Dataset
    print("\n📊 Creating dataset...")
    dataset = create_dataset(tokenizer, n_samples=200)
    print(f"   Samples: {len(dataset)}")
    
    # Training args
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Callbacks
    callbacks = []
    
    callbacks.append(GradienceCallback(
        out_dir=os.path.join(out_dir, "gradience"),
        guard_enabled=guard,
        guard_snapshot_interval=20,
        spectral_interval=10,
        telemetry_interval=5,
    ))
    
    if inject_nan:
        callbacks.append(nan_injector_callback(inject_step))
    
    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    
    print("\n🚀 Training...")
    t0 = time.time()
    
    try:
        trainer.train()
        success = True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        success = False
    
    duration = time.time() - t0
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Duration:  {duration:.1f}s")
    print(f"  Success:   {success}")
    
    summary_path = os.path.join(out_dir, "gradience", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        
        print(f"\n📈 Summary:")
        print(f"  Steps:     {summary.get('total_steps', 'N/A')}")
        print(f"  Loss:      {summary.get('final_loss', 'N/A')}")
        
        g = summary.get("guard", {})
        print(f"\n🛡️  Guard:")
        print(f"  Integrity: {g.get('integrity_status', 'N/A')}")
        print(f"  Rollbacks: {g.get('total_rollbacks', 0)}")
        print(f"  Recoveries: {g.get('total_recoveries', 0)}")
        
        s = summary.get("spectral", {})
        kappa = s.get("final_kappa")
        print(f"\n📊 Spectral:")
        print(f"  Final κ̃:  {kappa:.1f}" if kappa else "  Final κ̃:  N/A")
        print(f"  Risk:     {s.get('final_risk_level', 'N/A')}")
    
    print("\n📁 Files:")
    gdir = os.path.join(out_dir, "gradience")
    if os.path.exists(gdir):
        for f in sorted(os.listdir(gdir)):
            fp = os.path.join(gdir, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                print(f"  {f}: {sz:,} bytes")
    
    print("=" * 60)
    return success


def main():
    parser = argparse.ArgumentParser(description="Gradience Demo")
    parser.add_argument("--guard", action="store_true", default=True,
                       help="Enable guard (default: True)")
    parser.add_argument("--no-guard", action="store_true",
                       help="Disable guard")
    parser.add_argument("--inject-nan", action="store_true",
                       help="Inject NaN to test recovery")
    parser.add_argument("--inject-step", type=int, default=50,
                       help="Step for NaN injection (default: 50)")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Max training steps (default: 100)")
    parser.add_argument("--out-dir", type=str, default="./gradience_demo",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not check_deps():
        sys.exit(1)
    
    guard = not args.no_guard
    
    success = run_demo(
        guard=guard,
        inject_nan=args.inject_nan,
        inject_step=args.inject_step,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
