import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os

os.makedirs('./experiments/overfit', exist_ok=True)

device = torch.device('cuda')
torch.manual_seed(42)

# Tiny dataset (50 samples), big model
X = torch.randn(50, 64)
y = torch.randint(0, 10, (50,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=True)

# Test set (different samples)
X_test = torch.randn(200, 64)
y_test = torch.randint(0, 10, (200,))

# Overparameterized model (way more params than data)
model = nn.Sequential(
    nn.Linear(64, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: 50")
print(f"Overparameterization ratio: {sum(p.numel() for p in model.parameters()) / 50:.0f}x\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

from gradience.research import compute_full_spectrum, aggregate_layerwise_spectra

results = []
print("=== DELIBERATE OVERFITTING EXPERIMENT ===\n")
print(f"{'Step':>5} {'Train%':>8} {'Test%':>8} {'Loss':>8} {'κ':>10} {'Rank':>8}")
print("-" * 55)

for step in range(2001):
    # Train step
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            # Train acc
            train_pred = model(X.to(device)).argmax(1)
            train_acc = (train_pred == y.to(device)).float().mean().item()
            # Test acc
            test_pred = model(X_test.to(device)).argmax(1)
            test_acc = (test_pred == y_test.to(device)).float().mean().item()
        
        # Spectra
        spectra = [compute_full_spectrum(m.weight, f'layer_{i}', step) 
                   for i, m in enumerate(model) if isinstance(m, nn.Linear)]
        agg = aggregate_layerwise_spectra(spectra)
        
        results.append({
            'step': step,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss': loss.item(),
            'kappa': agg.get('mean_kappa'),
            'effective_rank': agg.get('mean_effective_rank'),
        })
        
        print(f"{step:>5} {train_acc*100:>7.1f}% {test_acc*100:>7.1f}% {loss.item():>8.4f} {agg.get('mean_kappa',0):>10.1f} {agg.get('mean_effective_rank',0):>8.1f}")

with open('./experiments/overfit/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to ./experiments/overfit/results.json")

# Analysis
print("\n=== MEMORIZATION SIGNATURE ===")
print(f"Final train acc: {results[-1]['train_acc']*100:.1f}%")
print(f"Final test acc: {results[-1]['test_acc']*100:.1f}%")
print(f"Generalization gap: {(results[-1]['train_acc'] - results[-1]['test_acc'])*100:.1f}%")
print(f"Rank trajectory: {results[0]['effective_rank']:.1f} → {results[-1]['effective_rank']:.1f}")
