import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os

os.makedirs('./experiments/cifar_resnet', exist_ok=True)

# Simple ResNet-18 style network
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10('./data', train=False, transform=transform_test)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

device = torch.device('cuda')
model = SmallResNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

from gradience.research import compute_full_spectrum

results = []
epochs = 50

print("=== CIFAR-10 ResNet Training ===\n")

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()
    
    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    test_acc = correct / total
    
    # Spectral metrics on fc layer
    with torch.no_grad():
        spec = compute_full_spectrum(model.fc.weight, 'fc', epoch)
    
    results.append({
        'epoch': epoch,
        'train_loss': train_loss / len(train_loader),
        'test_acc': test_acc,
        'kappa': spec.kappa,
        'effective_rank': spec.effective_rank,
    })
    
    print(f"Epoch {epoch:>3} | Loss: {train_loss/len(train_loader):.4f} | "
          f"Test: {test_acc*100:.1f}% | κ: {spec.kappa:.0f} | Rank: {spec.effective_rank:.1f}")

with open('./experiments/cifar_resnet/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to ./experiments/cifar_resnet/results.json")
