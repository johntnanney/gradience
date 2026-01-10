import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os

class ModularDataset(Dataset):
    def __init__(self, p, op, split, train_frac=0.3):
        self.p = p
        all_pairs = [(i,j) for i in range(p) for j in range(p)]
        n = int(len(all_pairs) * train_frac)
        self.pairs = all_pairs[:n] if split=='train' else all_pairs[n:]
        self.op = op
        
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, i):
        a, b = self.pairs[i]
        c = (a + b) % self.p  # ADDITION
        return {'input_ids': torch.tensor([a, b, self.p]), 'labels': torch.tensor(c)}

p = 97
device = torch.device('cuda')
os.makedirs('./experiments/grokking_add_wd', exist_ok=True)

train = ModularDataset(p, 'add', 'train')
test = ModularDataset(p, 'add', 'test')
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=256)

model = nn.Sequential(
    nn.Embedding(p+1, 128),
    nn.Flatten(),
    nn.Linear(128*3, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, p),
).to(device)

# SAME weight decay as multiplication
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

from gradience.research import compute_full_spectrum, aggregate_layerwise_spectra

results = []
step = 0
max_steps = 50000  # Should be enough if WD is the key

print("=== ADDITION with weight_decay=0.1 ===\n")

data_iter = iter(train_loader)
while step < max_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    
    x, y = batch['input_ids'].to(device), batch['labels'].to(device)
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for tb in test_loader:
                tx, ty = tb['input_ids'].to(device), tb['labels'].to(device)
                pred = model(tx).argmax(-1)
                correct += (pred == ty).sum().item()
                total += len(ty)
        test_acc = correct / total
        
        spectra = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                spectra.append(compute_full_spectrum(mod.weight, name, step))
        agg = aggregate_layerwise_spectra(spectra)
        
        results.append({
            'step': step,
            'loss': loss.item(),
            'test_acc': test_acc,
            'weight_mean_kappa': agg.get('mean_kappa'),
            'weight_mean_effective_rank': agg.get('mean_effective_rank'),
        })
        
        print(f"Step {step:>6} | Loss: {loss.item():.4f} | Test: {test_acc*100:.1f}% | Rank: {agg.get('mean_effective_rank', 0):.1f}")
        
        if test_acc > 0.95:
            print("\n🎉 GROKKING ACHIEVED!")
            break
        
        model.train()
    
    step += 1

with open('./experiments/grokking_add_wd/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to ./experiments/grokking_add_wd/results.json")
