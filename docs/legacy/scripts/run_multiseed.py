import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os

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
        c = (a * b) % self.p
        return torch.tensor([a, b, self.p]), torch.tensor(c)

def run_seed(seed):
    torch.manual_seed(seed)
    
    p = 97
    device = torch.device('cuda')
    
    train_set = ModularDataset(p, 'train')
    test_set = ModularDataset(p, 'test')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)
    
    model = nn.Sequential(
        nn.Embedding(p + 1, 128),
        nn.Flatten(),
        nn.Linear(128 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, p),
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    from gradience.research import compute_full_spectrum, aggregate_layerwise_spectra
    
    results = []
    step = 0
    max_steps = 40000
    
    data_iter = iter(train_loader)
    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for tx, ty in test_loader:
                    tx, ty = tx.to(device), ty.to(device)
                    correct += (model(tx).argmax(1) == ty).sum().item()
                    total += len(ty)
            test_acc = correct / total
            
            spectra = [compute_full_spectrum(m.weight, f'l{i}', step) 
                       for i, m in enumerate(model) if isinstance(m, nn.Linear)]
            agg = aggregate_layerwise_spectra(spectra)
            
            results.append({
                'step': step,
                'test_acc': test_acc,
                'kappa': agg.get('mean_kappa'),
                'effective_rank': agg.get('mean_effective_rank'),
            })
            
            if test_acc > 0.95:
                break
            
            model.train()
        
        step += 1
    
    return results

os.makedirs('./experiments/multiseed', exist_ok=True)

print("=== MULTI-SEED REPLICATION (5 seeds) ===\n")

all_results = {}
for seed in [42, 123, 456, 789, 1011]:
    print(f"Running seed {seed}...", end=" ", flush=True)
    results = run_seed(seed)
    all_results[seed] = results
    final_acc = results[-1]['test_acc']
    final_step = results[-1]['step']
    grokked = "✅" if final_acc > 0.95 else "❌"
    print(f"Final: {final_acc*100:.1f}% at step {final_step} {grokked}")

with open('./experiments/multiseed/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n=== SUMMARY ===")
grok_steps = []
for seed, res in all_results.items():
    if res[-1]['test_acc'] > 0.95:
        grok_step = res[-1]['step']
        grok_steps.append(grok_step)
        print(f"Seed {seed}: Grokked at step {grok_step}")
    else:
        print(f"Seed {seed}: Did not grok (final acc: {res[-1]['test_acc']*100:.1f}%)")

if grok_steps:
    print(f"\nGrok step: {min(grok_steps)} - {max(grok_steps)} (mean: {sum(grok_steps)/len(grok_steps):.0f})")
