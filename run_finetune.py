import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

os.makedirs('./experiments/finetune', exist_ok=True)

device = torch.device('cuda')

# Simple sentiment dataset (SST-2 style)
class SimpleClassificationDataset(Dataset):
    def __init__(self, tokenizer, split='train', n_samples=1000):
        # Synthetic data: positive/negative phrases
        positive = [
            "great movie", "loved it", "excellent film", "amazing story",
            "wonderful acting", "highly recommend", "fantastic", "brilliant",
            "masterpiece", "outstanding", "superb", "best ever", "perfect",
            "incredible", "beautiful", "touching", "inspiring", "delightful"
        ]
        negative = [
            "terrible movie", "hated it", "awful film", "boring story",
            "bad acting", "do not recommend", "horrible", "worst",
            "disaster", "disappointing", "waste of time", "painful",
            "dreadful", "unwatchable", "terrible", "stupid", "annoying"
        ]
        
        torch.manual_seed(42 if split == 'train' else 123)
        self.samples = []
        for _ in range(n_samples // 2):
            pos = positive[torch.randint(len(positive), (1,)).item()]
            neg = negative[torch.randint(len(negative), (1,)).item()]
            self.samples.append((pos, 1))
            self.samples.append((neg, 0))
        
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        text, label = self.samples[i]
        enc = self.tokenizer(text, padding='max_length', max_length=32, 
                            truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }

print("=== FINE-TUNING SPECTRAL ANALYSIS ===\n")
print("Loading model...")

# Use DistilBERT (small, fast)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)

# Freeze base, only train classifier (simulates LoRA-style)
for param in model.distilbert.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
    
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

train_set = SimpleClassificationDataset(tokenizer, 'train', n_samples=500)
test_set = SimpleClassificationDataset(tokenizer, 'test', n_samples=200)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
criterion = nn.CrossEntropyLoss()

from gradience.research import compute_full_spectrum

results = []
print(f"\n{'Step':>5} {'Train%':>8} {'Test%':>8} {'Loss':>8} {'κ':>10} {'Rank':>8}")
print("-" * 55)

step = 0
for epoch in range(20):
    model.train()
    epoch_loss = 0
    correct = total = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        preds = outputs.logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        step += 1
    
    train_acc = correct / total
    
    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            correct += (outputs.logits.argmax(-1) == labels).sum().item()
            total += len(labels)
    test_acc = correct / total
    
    # Spectral analysis of classifier head
    with torch.no_grad():
        # Classifier is typically: Linear(768, 768) + Linear(768, 2)
        spec = compute_full_spectrum(model.classifier.weight, 'classifier', epoch)
    
    results.append({
        'epoch': epoch,
        'step': step,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'loss': epoch_loss / len(train_loader),
        'kappa': spec.kappa,
        'effective_rank': spec.effective_rank,
    })
    
    print(f"{epoch:>5} {train_acc*100:>7.1f}% {test_acc*100:>7.1f}% {epoch_loss/len(train_loader):>8.4f} {spec.kappa:>10.1f} {spec.effective_rank:>8.1f}")

with open('./experiments/finetune/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== FINE-TUNING SIGNATURE ANALYSIS ===")
print(f"Initial κ: {results[0]['kappa']:.1f}")
print(f"Final κ: {results[-1]['kappa']:.1f}")
print(f"κ change: {results[-1]['kappa']/results[0]['kappa']:.2f}x")
print(f"\nInitial rank: {results[0]['effective_rank']:.2f}")
print(f"Final rank: {results[-1]['effective_rank']:.2f}")
print(f"\nConvergence: {results[-1]['test_acc']*100:.1f}% test accuracy")

# Detect equilibrium (when metrics stabilize)
kappas = [r['kappa'] for r in results]
for i in range(2, len(kappas)):
    recent_var = sum((k - sum(kappas[i-2:i+1])/3)**2 for k in kappas[i-2:i+1]) / 3
    if recent_var < 0.1 * kappas[i]**2:  # CV < 10%
        print(f"\nEquilibrium reached at epoch {i}")
        break
