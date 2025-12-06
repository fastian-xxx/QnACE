"""
Train emotion model on cleaned FER+ dataset with soft labels.

This uses:
1. FER+ majority-voted labels (cleaner than original FER2013)
2. Soft label training (knowledge from all annotators)
3. Heavy augmentation and modern techniques

Usage:
    python src/train_on_cleaned_data.py --data-dir data/combined --epochs 50
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


class CleanedFERDataset(Dataset):
    """Dataset for cleaned FER+ data with soft labels."""
    
    def __init__(self, csv_path: str, fer2013_path: str, transform=None, use_soft_labels=True):
        """
        Args:
            csv_path: Path to train.csv/val.csv/test.csv
            fer2013_path: Path to original fer2013.csv (for pixel data)
            transform: Image transforms
            use_soft_labels: Whether to return soft labels
        """
        self.transform = transform
        self.use_soft_labels = use_soft_labels
        self.samples = []
        
        # Load original FER2013 pixels
        fer_pixels = {}
        with open(fer2013_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                pixels = np.array([int(p) for p in row["pixels"].split()], dtype=np.uint8)
                fer_pixels[idx] = pixels.reshape(48, 48)
        
        # Load cleaned labels
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_idx = int(row["image_path"])  # We stored index as image_path
                label = int(row["emotion"])
                soft_labels = np.array([float(x) for x in row["soft_labels"].split(",")], dtype=np.float32)
                
                if img_idx in fer_pixels:
                    self.samples.append((fer_pixels[img_idx], label, soft_labels))
        
        print(f"Loaded {len(self.samples)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pixels, label, soft_labels = self.samples[idx]
        
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.use_soft_labels:
            return image, label, torch.tensor(soft_labels)
        return image, label


class SoftLabelLoss(nn.Module):
    """Cross entropy loss with soft labels (knowledge distillation style)."""
    
    def __init__(self, temperature=1.0, alpha=0.5):
        """
        Args:
            temperature: Softmax temperature for soft labels
            alpha: Weight for soft label loss (1-alpha for hard label loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, hard_labels, soft_labels):
        # Hard label loss
        hard_loss = self.ce_loss(logits, hard_labels)
        
        # Soft label loss (KL divergence)
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        soft_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class FocalSoftLoss(nn.Module):
    """Focal loss combined with soft labels."""
    
    def __init__(self, gamma=2.0, alpha=0.5, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights
    
    def forward(self, logits, hard_labels, soft_labels):
        # Soft label component
        log_probs = F.log_softmax(logits, dim=1)
        soft_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        
        # Focal loss component
        ce = F.cross_entropy(logits, hard_labels, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        return self.alpha * soft_loss + (1 - self.alpha) * focal_loss


def get_class_weights(dataset):
    """Compute class weights for imbalanced data."""
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


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels, soft_labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        soft_labels = soft_labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels, soft_labels)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / total, correct / total


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
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
    parser.add_argument("--data-dir", type=str, default="data/combined")
    parser.add_argument("--fer2013-path", type=str, default="data/fer2013.csv")
    parser.add_argument("--arch", type=str, default="efficientnet_b2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--soft-label-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="models")
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = CleanedFERDataset(
        os.path.join(args.data_dir, "train.csv"),
        args.fer2013_path,
        transform=get_transforms(train=True, img_size=args.img_size),
        use_soft_labels=True
    )
    val_dataset = CleanedFERDataset(
        os.path.join(args.data_dir, "val.csv"),
        args.fer2013_path,
        transform=get_transforms(train=False, img_size=args.img_size),
        use_soft_labels=True
    )
    test_dataset = CleanedFERDataset(
        os.path.join(args.data_dir, "test.csv"),
        args.fer2013_path,
        transform=get_transforms(train=False, img_size=args.img_size),
        use_soft_labels=True
    )
    
    # Class weights and sampling
    class_weights = get_class_weights(train_dataset).to(device)
    sample_weights = get_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Model
    model = build_model(args.arch).to(device)
    print(f"Model: {args.arch}")
    
    # Loss with soft labels
    criterion = FocalSoftLoss(gamma=2.0, alpha=args.soft_label_alpha, class_weights=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Soft label alpha: {args.soft_label_alpha}")
    
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.6f})")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        val_acc, val_per_class = validate(model, val_loader, device)
        print(f"  Val: acc={val_acc:.4f}")
        print(f"  Per-class: {dict(zip(EMOTION_LABELS, [f'{a:.3f}' for a in val_per_class]))}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "img_size": args.img_size,
            }, os.path.join(args.output_dir, "best_emotion_model_cleaned.pth"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
    
    # Final test
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_emotion_model_cleaned.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_acc, test_per_class = validate(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nPer-class Test Accuracy:")
    
    baseline = {"angry": 0.542, "disgust": 0.286, "fear": 0.250, "happy": 0.831, 
               "sad": 0.582, "surprise": 0.754, "neutral": 0.530}
    v1 = {"angry": 0.668, "disgust": 0.764, "fear": 0.506, "happy": 0.866,
          "sad": 0.519, "surprise": 0.846, "neutral": 0.720}
    
    for label, acc in zip(EMOTION_LABELS, test_per_class):
        b = baseline[label]
        v = v1[label]
        print(f"  {label:>8}: {acc:.3f} (baseline: {b:.3f}, v1: {v:.3f})")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

