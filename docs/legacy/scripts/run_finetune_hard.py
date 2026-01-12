import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import random

os.makedirs('./experiments/finetune_hard', exist_ok=True)
device = torch.device('cuda')

# Harder task: 4-class sentiment with subtle distinctions
class HardClassificationDataset(Dataset):
    def __init__(self, tokenizer, split='train', n_samples=400):
        # 4 classes: very negative, slightly negative, slightly positive, very positive
        templates = {
            0: ["terrible disaster", "absolutely horrible", "complete waste", "utterly awful", 
                "worst ever", "total failure", "extremely bad", "painfully boring"],
            1: ["somewhat disappointing", "not great", "could be better", "mediocre at best",
                "underwhelming", "forgettable", "nothing special", "mildly annoying"],
            2: ["fairly enjoyable", "decent enough", "reasonably good", "pleasant surprise",
                "worth watching", "mostly entertaining", "generally liked", "quite nice"],
            3: ["absolutely brilliant", "masterpiece", "outstanding", "exceptional",
                "truly amazing", "perfect", "incredible", "phenomenal"],
        }
        
        random.seed(42 if split == 'train' else 123)
        self.samples = []
        per_class = n_samples // 4
        for label, phrases in templates.items():
            for _ in range(per_class):
                phrase = random.choice(phrases)
                self.samples.append((phrase, label))
        
        random.shuffle(self.samples)
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

def run_experiment(name, lr, weight_decay, n_train):
    print(f"\n{'='*60}")
    print(f"{name}: LR={lr}, WD={weight_decay}, N={n_train}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4  # 4 classes now
    ).to(device)
    
    # Freeze base
    for param in model.distilbert.parameters():
        param.requires_grad = False
    
    train_set = HardClassificationDataset(tokenizer, 'train', n_samples=n_train)
    test_set = HardClassificationDataset(tokenizer, 'test', n_samples=200)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    results = []
    print(f"{'Ep':>3} {'Train%':>7} {'Test%':>7} {'Loss':>8} {'κ':>8} {'Rank':>6}")
    print("-" * 45)
    
    for epoch in range(30):
        model.train()
        epoch_loss = correct = total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            
            epoch_loss += outputs.loss.item()
            correct += (outputs.logits.argmax(-1) == batch['labels'].to(device)).sum().item()
            total += len(batch['labels'])
        
        train_acc = correct / total
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                correct += (outputs.logits.argmax(-1) == batch['labels'].to(device)).sum().item()
                total += len(batch['labels'])
        test_acc = correct / total
        
        spec = compute_full_spectrum(model.classifier.weight, 'clf', epoch)
        
        results.append({
            'epoch': epoch, 'train_acc': train_acc, 'test_acc': test_acc,
            'loss': epoch_loss / len(train_loader),
            'kappa': spec.kappa, 'effective_rank': spec.effective_rank,
        })
        
        print(f"{epoch:>3} {train_acc*100:>6.1f}% {test_acc*100:>6.1f}% "
              f"{epoch_loss/len(train_loader):>8.4f} {spec.kappa:>8.1f} {spec.effective_rank:>6.2f}")
    
    return results

all_results = {}

# Healthy baseline
all_results['healthy'] = run_experiment("Healthy", lr=2e-4, weight_decay=0.01, n_train=400)

# Memorization setup: tiny data, no regularization
all_results['memorize'] = run_experiment("Memorization risk", lr=5e-4, weight_decay=0.0, n_train=40)

# High LR
all_results['high_lr'] = run_experiment("High LR", lr=5e-3, weight_decay=0.01, n_train=400)

with open('./experiments/finetune_hard/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary
print("\n" + "="*70)
print("SUMMARY - 4-CLASS FINE-TUNING")
print("="*70)
print(f"\n{'Experiment':<20} {'Train%':>8} {'Test%':>8} {'Gap':>8} {'κ range':>12}")
print("-" * 60)
for name, res in all_results.items():
    final = res[-1]
    gap = (final['train_acc'] - final['test_acc']) * 100
    kappas = [r['kappa'] for r in res]
    print(f"{name:<20} {final['train_acc']*100:>7.1f}% {final['test_acc']*100:>7.1f}% "
          f"{gap:>+7.1f}% {min(kappas):.1f}-{max(kappas):.1f}")
