"""
Train emotion model on unified multi-dataset.

This script trains on the combined dataset from:
- FER2013 / FER+ 
- AffectNet
- RAF-DB
- CK+

Expected to achieve 85-92% accuracy with AffectNet + RAF-DB.

Usage:
    python src/train_unified.py --data-dir data/unified --epochs 50 --arch efficientnet_b2
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnifiedDataset(Dataset):
    """Dataset for unified multi-source emotion data."""
    
    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self.samples = []
        
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label = int(row["label"])
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_class_weights(dataset):
    """Compute class weights."""
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(dataset):
    """Get per-sample weights for balanced sampling."""
    labels = np.array([s[1] for s in dataset.samples])
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    return class_weights[labels]


def build_model(arch="efficientnet_b2"):
    """Build emotion classifier."""
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )
        return model
    elif arch == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )
        return model
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def get_transforms(train=True, img_size=260):
    """Get data transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_mixup=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply mixup with 50% probability
        if use_mixup and random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct/max(total, 1))
    
    return total_loss / len(loader.dataset), correct / max(total, 1)


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            for i in range(NUM_CLASSES):
                mask = labels == i
                per_class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                per_class_total[i] += mask.sum().item()
    
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    return correct / total, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/unified")
    parser.add_argument("--arch", type=str, default="efficientnet_b2",
                       choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_b4",
                               "convnext_tiny", "convnext_small"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Architecture: {args.arch}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = UnifiedDataset(
        os.path.join(args.data_dir, "train.csv"),
        transform=get_transforms(train=True, img_size=args.img_size)
    )
    val_dataset = UnifiedDataset(
        os.path.join(args.data_dir, "val.csv"),
        transform=get_transforms(train=False, img_size=args.img_size)
    )
    test_dataset = UnifiedDataset(
        os.path.join(args.data_dir, "test.csv"),
        transform=get_transforms(train=False, img_size=args.img_size)
    )
    
    # Class weights and balanced sampling
    class_weights = get_class_weights(train_dataset).to(device)
    sample_weights = get_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Model
    model = build_model(args.arch).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Resume if specified
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.6f})")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        val_acc, val_per_class = validate(model, val_loader, device)
        print(f"  Val: acc={val_acc:.4f}")
        print(f"  Per-class: {dict(zip(EMOTION_LABELS, [f'{a:.3f}' for a in val_per_class]))}")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "img_size": args.img_size,
            }, os.path.join(args.output_dir, "best_emotion_model_unified.pth"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
        
        # Save latest
        torch.save({
            "epoch": epoch + 1,
            "arch": args.arch,
            "model_state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "img_size": args.img_size,
        }, os.path.join(args.output_dir, "latest_emotion_model_unified.pth"))
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_emotion_model_unified.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_acc, test_per_class = validate(model, test_loader, device)
    
    print(f"\n🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nPer-class Test Accuracy:")
    
    # Compare with baselines
    baseline_fer = {"angry": 0.542, "disgust": 0.286, "fear": 0.250, "happy": 0.831, 
                   "sad": 0.582, "surprise": 0.754, "neutral": 0.530}
    v1_model = {"angry": 0.668, "disgust": 0.764, "fear": 0.506, "happy": 0.866,
                "sad": 0.519, "surprise": 0.846, "neutral": 0.720}
    
    for label, acc in zip(EMOTION_LABELS, test_per_class):
        b = baseline_fer[label]
        v1 = v1_model[label]
        improvement = (acc - b) * 100
        print(f"  {label:>8}: {acc:.3f} (baseline: {b:.3f}, v1: {v1:.3f}, improvement: {'+' if improvement >= 0 else ''}{improvement:.1f}%)")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_emotion_model_unified.pth')}")
    
    # Target assessment
    print("\n" + "="*60)
    print("TARGET ASSESSMENT")
    print("="*60)
    if test_acc >= 0.90:
        print("🎉 GOAL ACHIEVED! Accuracy >= 90%")
    elif test_acc >= 0.85:
        print("✓ Good progress! Accuracy >= 85%")
        print("  To reach 90%+, consider adding more data or using ensemble models.")
    elif test_acc >= 0.80:
        print("✓ Significant improvement! Accuracy >= 80%")
        print("  Add AffectNet (if not already) to push to 85-90%.")
    else:
        print("⚠️  Accuracy below 80%")
        print("  Ensure AffectNet and RAF-DB are properly integrated.")


if __name__ == "__main__":
    main()

