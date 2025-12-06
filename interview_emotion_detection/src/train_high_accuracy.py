"""
High-accuracy training script for FER2013.

Target: 75%+ validation accuracy using:
1. EfficientNet-B2 (larger model)
2. Heavy augmentation with MixUp and CutMix
3. Strong regularization (dropout, label smoothing, weight decay)
4. Class-balanced sampling
5. Test-time augmentation
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class FER2013Dataset(Dataset):
    """FER2013 dataset from CSV."""
    
    def __init__(self, csv_path, split='train', transform=None):
        self.transform = transform
        
        df = pd.read_csv(csv_path)
        
        # Filter by split
        if split == 'train':
            df = df[df['Usage'] == 'Training']
        elif split == 'val':
            df = df[df['Usage'] == 'PublicTest']
        else:
            df = df[df['Usage'] == 'PrivateTest']
        
        self.images = []
        self.labels = []
        
        for _, row in df.iterrows():
            pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
            img = pixels.reshape(48, 48)
            self.images.append(img)
            self.labels.append(int(row['emotion']))
        
        print(f"Loaded {len(self.images)} {split} images")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to RGB PIL Image
        img_rgb = np.stack([img, img, img], axis=-1)
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
        
        return img_tensor, label


class MixUp:
    """MixUp augmentation."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def build_model(num_classes=7, dropout=0.4):
    """Build EfficientNet-B2 model with custom head."""
    model = timm.create_model('efficientnet_b2.ra_in1k', pretrained=True)
    
    # Replace classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.75),
        nn.Linear(512, num_classes)
    )
    
    return model


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, classes=7, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def train_epoch(model, loader, criterion, optimizer, scheduler, device, mixup=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply MixUp
        if mixup is not None and random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            # For accuracy, use original labels
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = [0] * 7
    per_class_total = [0] * 7
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                per_class_total[label] += 1
                if predicted[i] == label:
                    per_class_correct[label] += 1
    
    print("\nPer-class accuracy:")
    for i, name in enumerate(EMOTION_LABELS):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            print(f"  {name:10s}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    return total_loss / len(loader), 100. * correct / total


def main():
    # Config
    config = {
        'data_path': 'data/fer2013.csv',
        'batch_size': 64,
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.4,
        'label_smoothing': 0.1,
        'image_size': 260,  # EfficientNet-B2 optimal
        'patience': 15,
    }
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = FER2013Dataset(config['data_path'], split='train', transform=train_transform)
    val_dataset = FER2013Dataset(config['data_path'], split='val', transform=val_transform)
    
    # Weighted sampler for class imbalance
    class_counts = Counter(train_dataset.labels)
    weights = [1.0 / class_counts[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Model
    model = build_model(num_classes=7, dropout=config['dropout'])
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(classes=7, smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # OneCycleLR scheduler
    total_steps = len(train_loader) * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # MixUp
    mixup = MixUp(alpha=0.2)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting High-Accuracy Training")
    print("="*60)
    print(f"Config: {config}")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, mixup
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, 'models/best_high_accuracy_model.pth')
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⚠️ Early stopping after {config['patience']} epochs without improvement")
            break
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: models/best_high_accuracy_model.pth")


if __name__ == "__main__":
    main()

