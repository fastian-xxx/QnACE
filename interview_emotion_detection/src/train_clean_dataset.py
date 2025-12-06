"""
Training Script for Clean Emotion Dataset
==========================================

Optimized for M1 Max and the cleaned dataset.
Target: 75-80%+ validation accuracy
"""

import os
import sys
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import numpy as np
import random

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class CleanEmotionDataset(Dataset):
    """Dataset loader for the cleaned dataset"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        for label, emotion in enumerate(EMOTION_NAMES):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                continue
            for img_path in emotion_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), label))
        
        print(f"Loaded {len(self.samples)} images from {self.root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


class EmotionModel(nn.Module):
    """EfficientNet-B2 based emotion classifier with better head"""
    
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        
        # Use EfficientNet-B2 for good accuracy/speed tradeoff
        self.backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 260, 260)
            features = self.backbone(dummy)
            feat_dim = features.shape[1]
        
        # Better classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_transforms(image_size=260, is_training=True):
    """Strong augmentation for training"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_mixup=True):
    """Train for one epoch with Mixup"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.random() > 0.5:
            # Apply mixup
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(EMOTION_NAMES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {name:10s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    # Configuration - Optimized for M1 Max 32-core GPU + 64GB RAM
    config = {
        'data_dir': 'data/clean_combined',
        'image_size': 224,  # Smaller image = less memory
        'batch_size': 128,  # Safe batch size for MPS
        'num_epochs': 30,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'dropout': 0.3,
        'num_workers': 0,
        'save_dir': 'models'
    }
    
    # Device setup - M1 Max 32-core GPU + 64GB
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("🚀 M1 Max 32-core GPU + 64GB RAM")
        print("   Batch size: 128 | Image size: 224")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Check if clean dataset exists
    clean_data_path = Path(config['data_dir'])
    if not clean_data_path.exists():
        print(f"❌ Clean dataset not found at {config['data_dir']}")
        print("   Run 'python src/clean_dataset.py' first!")
        sys.exit(1)
    
    # Data transforms
    train_transform = get_transforms(config['image_size'], is_training=True)
    val_transform = get_transforms(config['image_size'], is_training=False)
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading clean dataset...")
    print("="*60)
    
    train_dataset = CleanEmotionDataset(config['data_dir'], 'train', train_transform)
    val_dataset = CleanEmotionDataset(config['data_dir'], 'val', val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"\n📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"📦 Batches: {len(train_loader)} train, {len(val_loader)} val")
    print(f"⏱️  Expected: ~4-6 min per epoch\n")
    
    # Create model
    print("\n" + "="*60)
    print("Creating model: EfficientNet-B2")
    print("="*60)
    
    model = EmotionModel(num_classes=7, dropout=config['dropout']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting training on CLEAN dataset...")
    print("="*60)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_mixup=True
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = os.path.join(config['save_dir'], 'best_clean_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, save_path)
            print(f"  *** New best model! Val Acc: {val_acc:.2f}% ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping after {patience} epochs without improvement")
                break
        
        # Save latest
        save_path = os.path.join(config['save_dir'], 'latest_clean_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'config': config
        }, save_path)
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config['save_dir']}/best_clean_model.pth")


if __name__ == '__main__':
    main()

