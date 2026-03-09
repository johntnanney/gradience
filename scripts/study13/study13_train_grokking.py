#!/usr/bin/env python3
"""
Study 13: Grokking training with dense Hessian telemetry.

Trains an MLP on modular addition (mod 97) and logs curvature metrics
(λ₁, trace_H, gHg) every --hessian_every steps for DFA analysis.

Adapted from gradience/research/experiments.py::run_grokking_experiment()
with Hessian computation added and early stopping removed.

Usage:
    python study13_train_grokking.py --seed 42 --out_dir results/study13/seed_42
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ═══════════════════════════════════════════════════════════════════════════
# Dataset (from gradience/research/experiments.py)
# ═══════════════════════════════════════════════════════════════════════════


class ModularArithmeticDataset:
    """Dataset for modular addition (a + b) mod p."""

    def __init__(self, p=97, split="train", train_frac=0.3, seed=0):
        self.p = p
        # Generate all (a, b) pairs
        all_pairs = [(i, j) for i in range(p) for j in range(p)]

        # Deterministic shuffle for reproducible train/test split
        import random

        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        n_train = int(len(all_pairs) * train_frac)
        if split == "train":
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        c = (a + b) % self.p
        return {
            "input_ids": torch.tensor([a, b, self.p]),  # p is the "=" token
            "labels": torch.tensor(c),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Hessian utilities
# ═══════════════════════════════════════════════════════════════════════════


def _get_params(model):
    """Flatten model parameters into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])


def _get_grad(model):
    """Flatten model gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p.view(-1)))
    return torch.cat(grads)


def hvp(model, criterion, inputs, labels, v):
    """
    Hessian-vector product H·v via double backprop.

    Args:
        model: neural network
        criterion: loss function
        inputs, labels: single batch
        v: vector to multiply (same shape as flattened params)
    Returns:
        H·v as a flat tensor
    """
    model.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, labels)

    # First-order gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])

    # Gradient of (grad · v) w.r.t. parameters = H·v
    grad_v = torch.sum(flat_grad * v)
    hvp_grads = torch.autograd.grad(grad_v, model.parameters())
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])

    return flat_hvp.detach()


def top_eigenvalue(model, criterion, data_batches, max_iters=40, tol=1e-3):
    """
    Compute largest Hessian eigenvalue via power iteration.

    Averages HVP over multiple minibatches for stability.
    Returns: lambda_1 (float)
    """
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())

    # Random initial vector
    v = torch.randn(n_params, device=device)
    v = v / v.norm()

    lambda_prev = 0.0
    for iteration in range(max_iters):
        # Average HVP over batches
        Hv = torch.zeros(n_params, device=device)
        for inputs, labels in data_batches:
            Hv += hvp(model, criterion, inputs, labels, v)
        Hv /= len(data_batches)

        # Eigenvalue estimate
        lambda_1 = torch.dot(v, Hv).item()

        # Update eigenvector
        v_new = Hv / (Hv.norm() + 1e-10)

        # Convergence check
        if abs(lambda_1 - lambda_prev) / (abs(lambda_prev) + 1e-10) < tol:
            break
        lambda_prev = lambda_1
        v = v_new

    return lambda_1


def hutchinson_trace(model, criterion, data_batches, R=4):
    """
    Estimate Hessian trace via Hutchinson's stochastic estimator.

    Uses R Rademacher random vectors: trace(H) ≈ (1/R) Σ zᵀHz
    """
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())

    trace_est = 0.0
    for _ in range(R):
        # Rademacher random vector
        z = torch.randint(0, 2, (n_params,), device=device).float() * 2 - 1

        # Average HVP over batches
        Hz = torch.zeros(n_params, device=device)
        for inputs, labels in data_batches:
            Hz += hvp(model, criterion, inputs, labels, z)
        Hz /= len(data_batches)

        trace_est += torch.dot(z, Hz).item()

    return trace_est / R


def compute_gHg(model, criterion, data_batches):
    """
    Compute gᵀHg: gradient-Hessian-gradient product.

    This measures the curvature in the direction of the gradient.
    """
    device = next(model.parameters()).device

    # First compute the gradient (averaged over batches)
    model.zero_grad()
    total_loss = 0.0
    for inputs, labels in data_batches:
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        total_loss += loss.item()

    # Average gradients
    with torch.no_grad():
        g = _get_grad(model) / len(data_batches)

    g_norm = g.norm().item()
    if g_norm < 1e-12:
        return 0.0

    # Now compute Hg
    Hg = torch.zeros_like(g)
    for inputs, labels in data_batches:
        Hg += hvp(model, criterion, inputs, labels, g)
    Hg /= len(data_batches)

    gHg = torch.dot(g, Hg).item()
    return gHg


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Compute loss, accuracy on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    model.train()
    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Study 13: Grokking + DFA telemetry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/study13/seed_42")
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--hessian_every", type=int, default=50, help="Compute Hessian metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate val metrics every N steps")
    parser.add_argument("--p", type=int, default=97, help="Modulus for arithmetic")
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--M_batches", type=int, default=4, help="Number of batches for Hessian estimation")
    parser.add_argument("--hutch_R", type=int, default=4, help="Hutchinson trace estimator rank")
    parser.add_argument("--power_iters", type=int, default=40, help="Max power iterations for lambda_1")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print(f"  STUDY 13: Grokking DFA Telemetry  |  seed={args.seed}")
    print("=" * 65)
    print(f"  Task: addition mod {args.p}")
    print(f"  Train frac: {args.train_frac}")
    print(f"  Optimizer: AdamW lr={args.lr} wd={args.wd}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Hessian every: {args.hessian_every} steps")
    print(f"  Device: {device}")
    print()

    # ── Data ───────────────────────────────────────────────────────────
    train_dataset = ModularArithmeticDataset(p=args.p, split="train", train_frac=args.train_frac, seed=0)
    test_dataset = ModularArithmeticDataset(p=args.p, split="test", train_frac=args.train_frac, seed=0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print()

    # ── Model ──────────────────────────────────────────────────────────
    model = nn.Sequential(
        nn.Embedding(args.p + 1, 128),  # +1 for "=" token
        nn.Flatten(),
        nn.Linear(128 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, args.p),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # ── Save config ────────────────────────────────────────────────────
    config = vars(args)
    config["n_params"] = n_params
    config["n_train"] = len(train_dataset)
    config["n_test"] = len(test_dataset)
    config["device"] = str(device)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Training ───────────────────────────────────────────────────────
    telemetry_path = os.path.join(args.out_dir, "telemetry.jsonl")
    telemetry_file = open(telemetry_path, "w")

    data_iter = iter(train_loader)
    grokked = False
    grok_step = None
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward + backward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # ── Telemetry ──────────────────────────────────────────────
        if step % args.eval_every == 0:
            train_acc = (logits.detach().argmax(dim=-1) == labels).float().mean().item()
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            # Weight and gradient norms
            with torch.no_grad():
                weight_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            record = {
                "step": step,
                "train_loss": loss.item(),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "weight_norm": weight_norm,
                "grad_norm": grad_norm,
            }

            # ── Hessian metrics ────────────────────────────────────
            if step % args.hessian_every == 0:
                # Collect M minibatches for Hessian estimation
                hess_batches = []
                hess_iter = iter(train_loader)
                for _ in range(args.M_batches):
                    try:
                        hb = next(hess_iter)
                    except StopIteration:
                        hess_iter = iter(train_loader)
                        hb = next(hess_iter)
                    hess_batches.append((hb["input_ids"].to(device), hb["labels"].to(device)))

                model.train()  # ensure train mode for gradients

                # λ₁ via power iteration
                lambda1 = top_eigenvalue(model, criterion, hess_batches, max_iters=args.power_iters)
                record["lambda1"] = lambda1

                # trace(H) via Hutchinson
                trace_H = hutchinson_trace(model, criterion, hess_batches, R=args.hutch_R)
                record["trace_H"] = trace_H

                # gᵀHg
                gHg = compute_gHg(model, criterion, hess_batches)
                record["gHg"] = gHg

                record["hessian_computed"] = True
            else:
                record["hessian_computed"] = False

            # Write record
            telemetry_file.write(json.dumps(record) + "\n")
            telemetry_file.flush()

            # Grokking detection
            if not grokked and val_acc > 0.95:
                grokked = True
                grok_step = step
                print(f"\n  *** GROKKING at step {step} (val_acc={val_acc:.4f}) ***\n")

            # Progress
            if step % 1000 == 0:
                elapsed = time.time() - t_start
                status = f"GROKKED@{grok_step}" if grokked else "pre-grok"
                print(
                    f"  step {step:6d} | loss={loss.item():.6f} | "
                    f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
                    f"‖W‖={weight_norm:.1f} | {status} | {elapsed:.0f}s"
                )

    telemetry_file.close()

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = {
        "seed": args.seed,
        "grokked": grokked,
        "grok_step": grok_step,
        "final_val_acc": val_acc,
        "final_train_loss": loss.item(),
        "elapsed_seconds": elapsed,
        "n_telemetry_records": step // args.eval_every,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Done. Grokked: {grokked}" + (f" at step {grok_step}" if grokked else ""))
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Telemetry: {telemetry_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
