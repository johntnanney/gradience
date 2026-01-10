import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

os.makedirs('./experiments/finetune_pathological', exist_ok=True)
device = torch.device('cuda')

class SimpleClassificationDataset(Dataset):
    def __init__(self, tokenizer, split='train', n_samples=1000):
        positive = ["great movie", "loved it", "excellent film", "amazing story",
                    "wonderful acting", "highly recommend", "fantastic", "brilliant"]
        negative = ["terrible movie", "hated it", "awful film", "boring story",
                    "bad acting", "do not recommend", "horrible", "worst"]
        
        torch.manual_seed(42 if split == 'train' else 123)
        self.samples = []
        for _ in range(n_samples // 2):
            self.samples.append((positive[torch.randint(len(positive), (1,)).item()], 1))
            self.samples.append((negative[torch.randint(len(negative), (1,)).item()], 0))
        self.tokenizer = tokenizer
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, i):
        text, label = self.samples[i]
        enc = self.tokenizer(text, padding='max_length', max_length=32, 
                            truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }

from gradience.research import compute_full_spectrum

def run_finetuning(name, lr, weight_decay, unfreeze_base=False, n_train=500):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"LR={lr}, WD={weight_decay}, Unfreeze={unfreeze_base}, N={n_train}")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = model.to(device)
    
    # Freeze/unfreeze
    for param in model.distilbert.parameters():
        param.requires_grad = unfreeze_base
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    train_set = SimpleClassificationDataset(tokenizer, 'train', n_samples=n_train)
    test_set = SimpleClassificationDataset(tokenizer, 'test', n_samples=200)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay
    )
    
    results = []
    print(f"{'Ep':>3} {'Train%':>7} {'Test%':>7} {'Loss':>8} {'κ':>8} {'Rank':>6}")
    print("-" * 45)
    
    for epoch in range(20):
        model.train()
        epoch_loss = correct = total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            
            epoch_loss += outputs.loss.item()
            correct += (outputs.logits.argmax(-1) == labels).sum().item()
            total += len(labels)
        
        train_acc = correct / total
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch['input_ids'].to(device), 
                              attention_mask=batch['attention_mask'].to(device))
                correct += (outputs.logits.argmax(-1) == batch['labels'].to(device)).sum().item()
                total += len(batch['labels'])
        test_acc = correct / total
        
        spec = compute_full_spectrum(model.classifier.weight, 'clf', epoch)
        
        results.append({
            'epoch': epoch, 'train_acc': train_acc, 'test_acc': test_acc,
            'loss': epoch_loss / len(train_loader),
            'kappa': spec.kappa, 'effective_rank': spec.effective_rank,
        })
        
        print(f"{epoch:>3} {train_acc*100:>6.1f}% {test_acc*100:>6.1f}% {epoch_loss/len(train_loader):>8.4f} {spec.kappa:>8.1f} {spec.effective_rank:>6.2f}")
    
    return results

all_results = {}

# 1. Baseline (healthy) - already ran, but include for comparison
all_results['healthy'] = run_finetuning("Healthy (baseline)", lr=2e-4, weight_decay=0.01)

# 2. High LR (potential instability)
all_results['high_lr'] = run_finetuning("High LR", lr=1e-2, weight_decay=0.01)

# 3. No regularization + tiny data (memorization risk)
all_results['no_reg_tiny'] = run_finetuning("No reg + tiny data", lr=2e-4, weight_decay=0.0, n_train=50)

# 4. Full fine-tuning (all layers)
all_results['full_finetune'] = run_finetuning("Full fine-tune", lr=2e-5, weight_decay=0.01, unfreeze_base=True)

with open('./experiments/finetune_pathological/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"\n{'Experiment':<25} {'Final Test%':>12} {'κ range':>15} {'Signature':>15}")
print("-" * 70)
for name, res in all_results.items():
    kappas = [r['kappa'] for r in res]
    test = res[-1]['test_acc']
    k_range = f"{min(kappas):.1f}-{max(kappas):.1f}"
    
    # Classify signature
    k_growth = max(kappas) / min(kappas)
    if k_growth > 10:
        sig = "⚠️ UNSTABLE"
    elif k_growth > 3:
        sig = "⚠️ VOLATILE"
    else:
        sig = "✅ STABLE"
    
    print(f"{name:<25} {test*100:>11.1f}% {k_range:>15} {sig:>15}")
