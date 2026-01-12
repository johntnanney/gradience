import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import math

os.makedirs('./experiments/transformer_grok', exist_ok=True)

class ModularDataset(Dataset):
    def __init__(self, p, split, train_frac=0.3):
        self.p = p
        all_pairs = [(i, j) for i in range(p) for j in range(p)]
        n = int(len(all_pairs) * train_frac)
        self.pairs = all_pairs[:n] if split == 'train' else all_pairs[n:]
    
    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, i):
        a, b = self.pairs[i]
        c = (a * b) % self.p  # Multiplication
        # Format: [a, b, =] -> c
        return torch.tensor([a, b, self.p]), torch.tensor(c)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model)  # +1 for '=' token
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 
                                                    dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x) + self.pos
        x = self.transformer(x)
        return self.out(x[:, -1])  # Predict from last position

p = 97
device = torch.device('cuda')

train_set = ModularDataset(p, 'train')
test_set = ModularDataset(p, 'test')
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

model = TinyTransformer(p).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

from gradience.research import compute_full_spectrum, aggregate_layerwise_spectra

results = []
step = 0
max_steps = 50000

print("=== TRANSFORMER GROKKING (Multiplication mod 97) ===\n")
print(f"{'Step':>6} {'Loss':>8} {'Train%':>8} {'Test%':>8} {'κ':>10} {'Rank':>8}")
print("-" * 55)

data_iter = iter(train_loader)
while step < max_steps:
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        model.eval()
        
        # Train acc (sample)
        with torch.no_grad():
            train_correct = (model(x).argmax(1) == y).float().mean().item()
        
        # Test acc
        correct = total = 0
        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                pred = model(tx).argmax(1)
                correct += (pred == ty).sum().item()
                total += len(ty)
        test_acc = correct / total
        
        # Spectra from linear layers
        spectra = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[0] > 10:
                spectra.append(compute_full_spectrum(mod.weight, name, step))
        agg = aggregate_layerwise_spectra(spectra) if spectra else {}
        
        results.append({
            'step': step,
            'loss': loss.item(),
            'train_acc': train_correct,
            'test_acc': test_acc,
            'kappa': agg.get('mean_kappa', 0),
            'effective_rank': agg.get('mean_effective_rank', 0),
        })
        
        print(f"{step:>6} {loss.item():>8.4f} {train_correct*100:>7.1f}% {test_acc*100:>7.1f}% {agg.get('mean_kappa',0):>10.1f} {agg.get('mean_effective_rank',0):>8.1f}")
        
        if test_acc > 0.95:
            print("\n🎉 GROKKING ACHIEVED!")
            break
        
        model.train()
    
    step += 1

with open('./experiments/transformer_grok/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to ./experiments/transformer_grok/results.json")
