"""
Research Experiment Runner

Implements the experimental protocols from RESEARCH_AGENDA.md:

- Protocol A: Hessian-Weight Co-evolution Study
- Protocol B: Grokking Phase Transition Study
- Protocol D: Rank Evolution Survey

Run with:
    python -m gradience.research.experiments --protocol rank_evolution
    python -m gradience.research.experiments --protocol hessian_coevolution
    python -m gradience.research.experiments --protocol grokking

Requires: torch, transformers
"""

from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


def check_dependencies():
    """Check that required dependencies are available."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch transformers")
        return False


# ============================================================================
# Protocol D: Rank Evolution Survey
# ============================================================================

def run_rank_evolution_experiment(
    out_dir: str = "./experiments/rank_evolution",
    model_type: str = "gpt2-tiny",
    max_steps: int = 2000,
    measure_interval: int = 20,
    seed: int = 42,
):
    """
    Track effective rank evolution throughout training.
    
    Research questions:
    - How does effective rank evolve during training?
    - Is there a characteristic trajectory (expansion → compression)?
    - Do instabilities correlate with rank dynamics?
    """
    import torch
    from gradience.research import (
        compute_layerwise_spectra,
        aggregate_layerwise_spectra,
        RankTracker,
    )
    
    print("=" * 60)
    print("PROTOCOL D: Rank Evolution Survey")
    print("=" * 60)
    
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup model and data
    model, train_loader, _ = setup_experiment(model_type, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Trackers
    rank_tracker = RankTracker()
    results = []
    
    print(f"\nModel: {model_type}")
    print(f"Device: {device}")
    print(f"Steps: {max_steps}")
    print(f"Measure interval: {measure_interval}")
    print()
    
    step = 0
    data_iter = iter(train_loader)
    
    while step < max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Training step
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Measure spectra
        if step % measure_interval == 0:
            with torch.no_grad():
                spectra = compute_layerwise_spectra(model, step=step)
                agg = aggregate_layerwise_spectra(spectra)
                
                # Record per-layer ranks
                for s in spectra:
                    rank_tracker.add(
                        step=step,
                        effective_rank=s.effective_rank,
                        stable_rank=s.stable_rank,
                        numerical_rank=s.numerical_rank,
                        layer_name=s.layer_name,
                    )
                
                result = {
                    "step": step,
                    "loss": loss.item(),
                    "timestamp": time.time(),
                    **agg,
                    "rank_phase": rank_tracker.detect_rank_phase(),
                    "rank_velocity": rank_tracker.compute_rank_velocity(),
                    "rank_volatility": rank_tracker.compute_rank_volatility(),
                }
                results.append(result)
                
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                      f"EffRank: {agg.get('mean_effective_rank', 0):.1f} | "
                      f"κ: {agg.get('max_kappa', 0):.1f} | "
                      f"Phase: {rank_tracker.detect_rank_phase()}")
        
        step += 1
    
    # Save results
    results_path = os.path.join(out_dir, "rank_evolution_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    return results


# ============================================================================
# Protocol A: Hessian-Weight Co-evolution Study
# ============================================================================

def run_hessian_coevolution_experiment(
    out_dir: str = "./experiments/hessian_coevolution",
    model_type: str = "gpt2-tiny",
    max_steps: int = 1000,
    measure_interval: int = 50,
    n_hessian_eigenvalues: int = 5,
    n_hutchinson_samples: int = 30,
    seed: int = 42,
):
    """
    Track co-evolution of weight spectra and Hessian spectra.
    
    Research questions:
    - Does weight spectrum predict Hessian spectral properties?
    - At instability, which changes first?
    - What's the cross-correlation structure?
    """
    import torch
    from gradience.research import (
        compute_layerwise_spectra,
        aggregate_layerwise_spectra,
        compute_hessian_snapshot,
        create_loss_fn_for_batch,
        HessianTracker,
    )
    
    print("=" * 60)
    print("PROTOCOL A: Hessian-Weight Co-evolution Study")
    print("=" * 60)
    
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup
    model, train_loader, _ = setup_experiment(model_type, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    hessian_tracker = HessianTracker()
    results = []
    
    print(f"\nModel: {model_type}")
    print(f"Device: {device}")
    print(f"Steps: {max_steps}")
    print(f"Hessian eigenvalues: {n_hessian_eigenvalues}")
    print(f"NOTE: Hessian computation is expensive, expect ~10s per measurement")
    print()
    
    step = 0
    data_iter = iter(train_loader)
    
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Training step
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Measure
        if step % measure_interval == 0:
            print(f"Step {step}: Computing spectra...", end=" ", flush=True)
            
            # Weight spectra
            with torch.no_grad():
                spectra = compute_layerwise_spectra(model, step=step)
                weight_agg = aggregate_layerwise_spectra(spectra)
            
            # Hessian spectra (expensive)
            print("Hessian...", end=" ", flush=True)
            
            def loss_fn():
                out = model(**batch)
                return out.loss
            
            hessian_snap = compute_hessian_snapshot(
                model, loss_fn, step=step,
                n_top_eigenvalues=n_hessian_eigenvalues,
                n_trace_samples=n_hutchinson_samples,
            )
            hessian_tracker.add(hessian_snap)
            
            result = {
                "step": step,
                "loss": loss.item(),
                "timestamp": time.time(),
                # Weight spectra
                "weight_mean_kappa": weight_agg.get("mean_kappa"),
                "weight_max_kappa": weight_agg.get("max_kappa"),
                "weight_mean_effective_rank": weight_agg.get("mean_effective_rank"),
                # Hessian spectra
                "hessian_lambda_max": hessian_snap.lambda_max,
                "hessian_trace": hessian_snap.trace,
                "hessian_top_eigenvalues": hessian_snap.top_eigenvalues,
            }
            results.append(result)
            
            print(f"Done. Loss: {loss.item():.4f} | "
                  f"W-κ: {weight_agg.get('max_kappa', 0):.1f} | "
                  f"H-λmax: {hessian_snap.lambda_max:.2e}")
        
        step += 1
    
    # Save
    results_path = os.path.join(out_dir, "hessian_coevolution_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Compute cross-correlation
    print("\nComputing cross-correlations...")
    weight_kappas = [r["weight_max_kappa"] for r in results]
    hessian_lambdas = [r["hessian_lambda_max"] for r in results]
    
    if len(weight_kappas) > 10:
        from gradience.research import compute_autocorrelation
        
        # Normalize
        wk_mean = sum(weight_kappas) / len(weight_kappas)
        hl_mean = sum(hessian_lambdas) / len(hessian_lambdas)
        wk_norm = [w - wk_mean for w in weight_kappas]
        hl_norm = [h - hl_mean for h in hessian_lambdas]
        
        # Cross-correlation at various lags
        max_lag = min(20, len(results) // 2)
        xcorr = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                w = wk_norm[:lag]
                h = hl_norm[-lag:]
            elif lag > 0:
                w = wk_norm[lag:]
                h = hl_norm[:-lag]
            else:
                w, h = wk_norm, hl_norm
            
            if len(w) > 0:
                corr = sum(wi * hi for wi, hi in zip(w, h)) / len(w)
                xcorr.append({"lag": lag, "correlation": corr})
        
        xcorr_path = os.path.join(out_dir, "weight_hessian_xcorr.json")
        with open(xcorr_path, "w") as f:
            json.dump(xcorr, f, indent=2)
        print(f"Cross-correlation saved to {xcorr_path}")
    
    print(f"\nResults saved to {results_path}")
    return results


# ============================================================================
# Protocol B: Grokking Study
# ============================================================================

class ModularArithmeticDataset:
    """Dataset for modular arithmetic (grokking experiments)."""
    
    def __init__(self, p: int = 97, operation: str = "add", split: str = "train", train_frac: float = 0.3):
        import torch
        self.p = p
        self.operation = operation
        self._torch = torch
        
        # Generate all pairs
        all_pairs = [(i, j) for i in range(p) for j in range(p)]
        
        # Split
        n_train = int(len(all_pairs) * train_frac)
        if split == "train":
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        torch = self._torch
        a, b = self.pairs[idx]
        
        if self.operation == "add":
            c = (a + b) % self.p
        elif self.operation == "mult":
            c = (a * b) % self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
        
        # Encode as tokens: [a, b, =, c]
        # Using simple encoding: token_id = value
        return {
            "input_ids": torch.tensor([a, b, self.p]),  # p is the "=" token
            "labels": torch.tensor(c),
        }


def run_grokking_experiment(
    out_dir: str = "./experiments/grokking",
    p: int = 97,
    operation: str = "add",
    max_steps: int = 50000,
    measure_interval: int = 100,
    seed: int = 42,
):
    """
    Study grokking phenomenon with spectral monitoring.
    
    Research questions:
    - Can we detect precursors to the grokking transition?
    - Is there critical slowing down before grokking?
    - What do spectral metrics show at the transition?
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from gradience.research import (
        compute_layerwise_spectra,
        aggregate_layerwise_spectra,
        PhaseTransitionTracker,
        GrokDetector,
    )
    
    print("=" * 60)
    print("PROTOCOL B: Grokking Phase Transition Study")
    print("=" * 60)
    
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    # Create modular arithmetic dataset
    train_dataset = ModularArithmeticDataset(p=p, operation=operation, split="train")
    test_dataset = ModularArithmeticDataset(p=p, operation=operation, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Simple MLP for this task
    model = nn.Sequential(
        nn.Embedding(p + 1, 128),  # +1 for "=" token
        nn.Flatten(),
        nn.Linear(128 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, p),  # Output classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Trackers
    loss_tracker = PhaseTransitionTracker(observable_name="train_loss")
    acc_tracker = PhaseTransitionTracker(observable_name="test_acc")
    grok_detector = GrokDetector()
    
    results = []
    
    print(f"\nTask: {operation} mod {p}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Max steps: {max_steps}")
    print()
    
    step = 0
    data_iter = iter(train_loader)
    
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Training step
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        loss_tracker.add(step, loss.item())
        
        # Evaluate and measure
        if step % measure_interval == 0:
            model.eval()
            
            # Test accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for test_batch in test_loader:
                    test_ids = test_batch["input_ids"].to(device)
                    test_labels = test_batch["labels"].to(device)
                    test_logits = model(test_ids)
                    preds = test_logits.argmax(dim=-1)
                    correct += (preds == test_labels).sum().item()
                    total += len(test_labels)
            
            test_acc = correct / total
            acc_tracker.add(step, test_acc)
            
            # Train accuracy
            train_correct = (logits.argmax(dim=-1) == labels).float().mean().item()
            
            grok_detector.add(step, train_correct, test_acc)
            
            # Spectral metrics (on the linear layers)
            spectra = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    from gradience.research import compute_full_spectrum
                    spec = compute_full_spectrum(module.weight, layer_name=name, step=step)
                    spectra.append(spec)
            
            weight_agg = aggregate_layerwise_spectra(spectra) if spectra else {}
            
            # Phase transition metrics
            loss_metrics = loss_tracker.compute_metrics()
            acc_metrics = acc_tracker.compute_metrics()
            grok_status = grok_detector.detect_grokking()
            
            result = {
                "step": step,
                "train_loss": loss.item(),
                "train_acc": train_correct,
                "test_acc": test_acc,
                "timestamp": time.time(),
                # Spectral
                "weight_mean_kappa": weight_agg.get("mean_kappa"),
                "weight_mean_effective_rank": weight_agg.get("mean_effective_rank"),
                # Phase transition
                "loss_autocorr_time": loss_metrics.autocorr_time,
                "loss_variance_ratio": loss_metrics.variance_ratio,
                "loss_phase": loss_metrics.phase,
                "acc_autocorr_time": acc_metrics.autocorr_time,
                "acc_phase": acc_metrics.phase,
                # Grokking
                "grok_phase": grok_status["phase"],
                "generalization_gap": grok_status["generalization_gap"],
            }
            results.append(result)
            
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                  f"Train: {train_correct:.2f} | Test: {test_acc:.2f} | "
                  f"Grok: {grok_status['phase']}")
            
            # Early stopping if grokking achieved
            if test_acc > 0.99:
                print("\n🎉 Grokking achieved!")
                break
            
            model.train()
        
        step += 1
    
    # Save
    results_path = os.path.join(out_dir, "grokking_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    return results


# ============================================================================
# Helpers
# ============================================================================

def setup_experiment(model_type: str, seed: int):
    """Set up model and data for experiments."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    
    if model_type == "gpt2-tiny":
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        
        config = GPT2Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=4,
            n_head=4,
            n_positions=64,
        )
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Synthetic dataset
        class SyntheticLMDataset(Dataset):
            def __init__(self, vocab_size, seq_len, n_samples):
                self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                ids = self.data[idx]
                return {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone(),
                }
        
        train_dataset = SyntheticLMDataset(config.vocab_size, 64, 1000)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        return model, train_loader, tokenizer
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Gradience Research Experiments")
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["rank_evolution", "hessian_coevolution", "grokking"],
        help="Experiment protocol to run",
    )
    parser.add_argument("--out-dir", type=str, default="./experiments")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.protocol == "rank_evolution":
        run_rank_evolution_experiment(
            out_dir=os.path.join(args.out_dir, "rank_evolution"),
            max_steps=args.max_steps or 2000,
            seed=args.seed,
        )
    
    elif args.protocol == "hessian_coevolution":
        run_hessian_coevolution_experiment(
            out_dir=os.path.join(args.out_dir, "hessian_coevolution"),
            max_steps=args.max_steps or 1000,
            seed=args.seed,
        )
    
    elif args.protocol == "grokking":
        run_grokking_experiment(
            out_dir=os.path.join(args.out_dir, "grokking"),
            max_steps=args.max_steps or 50000,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
