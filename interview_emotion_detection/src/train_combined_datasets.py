"""
Advanced Training Script for Combined Emotion Datasets
Target: 90%+ accuracy using ~200K images from multiple sources

Datasets:
- FER2013 + FER+ (cleaned labels): ~28K train
- Kaggle FER: ~29K train + 7K val
- Emotion Detection: ~29K train + 7K test
- MMAFEDB: ~93K train + 17K val + 17K test

Total: ~150K training images, ~31K validation images
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
import timm
from tqdm import tqdm

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Emotion mapping (standardized across all datasets)
EMOTION_MAP = {
    'angry': 0, 'anger': 0,
    'disgust': 1, 'disgusted': 1,
    'fear': 2, 'fearful': 2,
    'happy': 3, 'happiness': 3,
    'neutral': 4,
    'sad': 5, 'sadness': 5,
    'surprise': 6, 'surprised': 6
}

EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class ImageFolderDataset(Dataset):
    """Generic dataset for loading images from folder structure"""
    
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Find the correct directory
        if (self.root_dir / split).exists():
            data_dir = self.root_dir / split
        elif (self.root_dir / 'images' / split).exists():
            data_dir = self.root_dir / 'images' / split
        else:
            data_dir = self.root_dir
        
        # Load all images
        for emotion_dir in data_dir.iterdir():
            if emotion_dir.is_dir():
                emotion_name = emotion_dir.name.lower()
                if emotion_name in EMOTION_MAP:
                    label = EMOTION_MAP[emotion_name]
                    for img_path in emotion_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            self.samples.append((str(img_path), label))
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a random valid sample if this one fails
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


class FER2013Dataset(Dataset):
    """Dataset for FER2013 CSV format with FER+ cleaned labels"""
    
    def __init__(self, csv_path, fer_plus_path=None, transform=None, split='train'):
        import pandas as pd
        
        self.transform = transform
        self.samples = []
        
        # Load FER2013
        df = pd.read_csv(csv_path)
        
        # Load FER+ labels if available
        fer_plus_labels = None
        if fer_plus_path and os.path.exists(fer_plus_path):
            fer_plus_df = pd.read_csv(fer_plus_path)
            fer_plus_labels = fer_plus_df.values
            print(f"Using FER+ cleaned labels")
        
        # Filter by usage
        usage_map = {'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}
        target_usage = usage_map.get(split, 'Training')
        
        for idx, row in df.iterrows():
            if row['Usage'] == target_usage:
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                pixels = pixels.reshape(48, 48)
                
                # Use FER+ label if available (majority voting)
                if fer_plus_labels is not None and idx < len(fer_plus_labels):
                    # FER+ format: columns 2-9 are emotion votes
                    votes = fer_plus_labels[idx, 2:9].astype(float)
                    if votes.sum() > 0:
                        label = int(np.argmax(votes))
                        # Skip low-agreement samples
                        if votes[label] / votes.sum() < 0.4:
                            continue
                    else:
                        label = int(row['emotion'])
                else:
                    label = int(row['emotion'])
                
                self.samples.append((pixels, label))
        
        print(f"Loaded {len(self.samples)} images from FER2013 ({split})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pixels, label = self.samples[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(pixels).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EmotionModel(nn.Module):
    """EfficientNet-based emotion classifier"""
    
    def __init__(self, num_classes=7, model_name='efficientnet_b2', pretrained=True, dropout=0.4):
        super().__init__()
        
        # Use timm for more model options
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feat_dim = features.shape[1]
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_transforms(image_size=224, is_training=True):
    """Get data augmentation transforms"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_combined_dataset(data_dir, transform, split='train'):
    """Create combined dataset from all sources"""
    
    datasets_list = []
    data_dir = Path(data_dir)
    
    # 1. FER2013 with FER+ labels
    fer2013_path = data_dir / 'fer2013.csv'
    fer_plus_path = data_dir / 'fer2013new.csv'
    if fer2013_path.exists():
        fer_split = 'train' if split == 'train' else 'val'
        fer_dataset = FER2013Dataset(fer2013_path, fer_plus_path, transform, fer_split)
        datasets_list.append(fer_dataset)
    
    # 2. Kaggle FER dataset
    kaggle_fer_dir = data_dir / 'kaggle_fer' / 'images'
    if kaggle_fer_dir.exists():
        kaggle_split = 'train' if split == 'train' else 'validation'
        kaggle_dataset = ImageFolderDataset(kaggle_fer_dir, transform, kaggle_split)
        datasets_list.append(kaggle_dataset)
    
    # 3. Emotion Detection dataset
    emotion_det_dir = data_dir / 'emotion_detection'
    if emotion_det_dir.exists():
        ed_split = 'train' if split == 'train' else 'test'
        ed_dataset = ImageFolderDataset(emotion_det_dir, transform, ed_split)
        datasets_list.append(ed_dataset)
    
    # 4. MMAFEDB dataset
    mmafedb_dir = data_dir / 'mmafedb' / 'MMAFEDB'
    if mmafedb_dir.exists():
        mma_split = 'train' if split == 'train' else 'valid'
        mma_dataset = ImageFolderDataset(mmafedb_dir, transform, mma_split)
        datasets_list.append(mma_dataset)
    
    if not datasets_list:
        raise ValueError(f"No datasets found in {data_dir}")
    
    combined = ConcatDataset(datasets_list)
    print(f"\nCombined {split} dataset: {len(combined)} total images")
    
    return combined


def get_class_weights(dataset):
    """Calculate class weights for balanced training"""
    
    class_counts = defaultdict(int)
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label] += 1
    
    total = sum(class_counts.values())
    weights = []
    
    for i in range(7):
        count = class_counts.get(i, 1)
        weight = total / (7 * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Step scheduler per batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
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
    # Configuration - OPTIMIZED FOR M1 MAX
    config = {
        'data_dir': 'data',
        'model_name': 'efficientnet_b2',  # Good balance of speed/accuracy
        'image_size': 224,
        'batch_size': 128,  # Larger batch for M1 Max (32GB unified memory)
        'num_epochs': 50,   # More epochs with better LR schedule
        'learning_rate': 3e-4,  # Higher starting LR
        'min_lr': 1e-5,     # Don't go below this
        'weight_decay': 1e-4,
        'dropout': 0.3,     # Slightly less dropout to reduce underfitting
        'num_workers': 0,   # 0 is faster on macOS with MPS!
        'save_dir': 'models'
    }
    
    # Device setup - Optimized for M1 Max
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        # Enable MPS fallback for unsupported ops
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit
        print("🚀 Using Apple Silicon MPS (M1 Max optimized)")
        print("   - Batch size: 128")
        print("   - num_workers: 0 (fastest for MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Data transforms
    train_transform = get_transforms(config['image_size'], is_training=True)
    val_transform = get_transforms(config['image_size'], is_training=False)
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_dataset = create_combined_dataset(config['data_dir'], train_transform, 'train')
    val_dataset = create_combined_dataset(config['data_dir'], val_transform, 'val')
    
    # Create dataloaders - Optimized for MPS
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,  # Not needed for MPS
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    # Create model
    print("\n" + "="*60)
    print(f"Creating model: {config['model_name']}")
    print("="*60)
    
    model = EmotionModel(
        num_classes=7,
        model_name=config['model_name'],
        pretrained=True,
        dropout=config['dropout']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function - CrossEntropy with Label Smoothing works better than Focal Loss
    # Label smoothing helps with noisy labels in emotion datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - OneCycleLR is MUCH better for reaching high accuracy
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # start_lr = max_lr / 10
        final_div_factor=100  # end_lr = max_lr / 1000
    )
    
    # Mixed precision - not needed for MPS (it handles this automatically)
    scaler = None
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train (pass scheduler for OneCycleLR per-batch stepping)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # No need to step scheduler here - OneCycleLR steps per batch
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            save_path = os.path.join(config['save_dir'], 'best_combined_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, save_path)
            print(f"  *** New best model saved! (Val Acc: {val_acc:.2f}%) ***")
        
        # Save latest model
        save_path = os.path.join(config['save_dir'], 'latest_combined_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config
        }, save_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {config['save_dir']}/best_combined_model.pth")


if __name__ == '__main__':
    main()

