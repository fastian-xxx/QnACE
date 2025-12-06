"""
Fine-tune an emotion recognition model on FER2013.

This script:
1. Loads FER2013 from the single CSV (fer2013.csv).
2. Builds an EfficientNet-B0 based classifier with class-weighted loss.
3. Applies heavy augmentation to improve generalization.
4. Trains and saves the best checkpoint.

Usage:
    python src/train_emotion_model.py --data-path data/fer2013.csv --epochs 30 --batch-size 64
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

# Emotion labels in FER2013 order
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTION_LABELS)


class FER2013Dataset(Dataset):
    """Custom dataset for FER2013 CSV."""

    def __init__(self, csv_path: str, split: str = "Training", transform=None):
        """
        Args:
            csv_path: Path to fer2013.csv
            split: One of "Training", "PublicTest", "PrivateTest"
            transform: Torchvision transforms
        """
        self.transform = transform
        self.samples = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] == split:
                    pixels = np.array([int(p) for p in row["pixels"].split()], dtype=np.uint8)
                    pixels = pixels.reshape(48, 48)
                    label = int(row["emotion"])
                    self.samples.append((pixels, label))

        print(f"Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pixels, label = self.samples[idx]
        # Convert to PIL for transforms
        image = Image.fromarray(pixels, mode="L")
        # Convert grayscale to RGB (3 channels) for pretrained models
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_class_weights(dataset: FER2013Dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(dataset: FER2013Dataset) -> np.ndarray:
    """Compute per-sample weights for WeightedRandomSampler."""
    labels = np.array([s[1] for s in dataset.samples])
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    return sample_weights


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 based emotion classifier."""
    # Use EfficientNet-B0 as backbone
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def get_transforms(train: bool = True):
    """Get data transforms with augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

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

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            for i in range(NUM_CLASSES):
                mask = labels == i
                per_class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                per_class_total[i] += mask.sum().item()

    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

    return total_loss / total, correct / total, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune emotion model on FER2013")
    parser.add_argument("--data-path", type=str, default="data/fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    train_dataset = FER2013Dataset(args.data_path, split="Training", transform=get_transforms(train=True))
    val_dataset = FER2013Dataset(args.data_path, split="PublicTest", transform=get_transforms(train=False))
    test_dataset = FER2013Dataset(args.data_path, split="PrivateTest", transform=get_transforms(train=False))

    # Compute class weights for loss
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Compute sample weights for balanced sampling
    sample_weights = get_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.3f}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc, val_per_class = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Per-class Val Acc: {dict(zip(EMOTION_LABELS, [f'{a:.3f}' for a in val_per_class]))}")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(args.output_dir) / "best_emotion_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_weights": class_weights.cpu(),
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

        # Save latest checkpoint
        latest_path = Path(args.output_dir) / "latest_emotion_model.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_weights": class_weights.cpu(),
        }, latest_path)

    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load(Path(args.output_dir) / "best_emotion_model.pth", map_location=device)["model_state_dict"])
    test_loss, test_acc, test_per_class = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Per-class Test Acc:")
    for label, acc in zip(EMOTION_LABELS, test_per_class):
        print(f"  {label:>8}: {acc:.3f}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {Path(args.output_dir) / 'best_emotion_model.pth'}")


if __name__ == "__main__":
    main()

