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
    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, i):
        a, b = self.pairs[i]
        c = (a * b) % self.p
        return torch.tensor([a, b, self.p]), torch.tensor(c)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 
                                                    dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x) + self.pos
        x = self.transformer(x)
        return self.out(x[:, -1])

from gradience.research import compute_full_spectrum, aggregate_layerwise_spectra

def run_seed(seed, max_steps=50000):
    torch.manual_seed(seed)
    p = 97
    device = torch.device('cuda')
    
    train_loader = DataLoader(ModularDataset(p, 'train'), batch_size=64, shuffle=True)
    test_loader = DataLoader(ModularDataset(p, 'test'), batch_size=256)
    
    model = TinyTransformer(p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    step = 0
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
        
        if step % 500 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for tx, ty in test_loader:
                    tx, ty = tx.to(device), ty.to(device)
                    correct += (model(tx).argmax(1) == ty).sum().item()
                    total += len(ty)
            test_acc = correct / total
            
            spectra = [compute_full_spectrum(m.weight, n, step) 
                       for n, m in model.named_modules() 
                       if isinstance(m, nn.Linear) and m.weight.shape[0] > 10]
            agg = aggregate_layerwise_spectra(spectra) if spectra else {}
            
            results.append({
                'step': step,
                'test_acc': test_acc,
                'kappa': agg.get('mean_kappa', 0),
                'effective_rank': agg.get('mean_effective_rank', 0),
            })
            
            if test_acc > 0.95:
                return results, step
            
            model.train()
        step += 1
    
    return results, None

os.makedirs('./experiments/transformer_multiseed', exist_ok=True)

print("=== TRANSFORMER MULTI-SEED (5 seeds) ===\n")
all_results = {}
grok_steps = []

for seed in [42, 123, 456, 789, 1011]:
    print(f"Seed {seed}...", end=" ", flush=True)
    results, grok_step = run_seed(seed)
    all_results[seed] = results
    
    if grok_step:
        grok_steps.append(grok_step)
        print(f"✅ Grokked at step {grok_step}")
    else:
        print(f"❌ Did not grok (final: {results[-1]['test_acc']*100:.1f}%)")

with open('./experiments/transformer_multiseed/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"Grokked: {len(grok_steps)}/5 seeds")
if grok_steps:
    print(f"Grok step: {min(grok_steps)} - {max(grok_steps)} (mean: {sum(grok_steps)/len(grok_steps):.0f})")
