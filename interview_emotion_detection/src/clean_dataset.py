"""
Dataset Cleaning Script for Emotion Recognition
================================================

This script cleans the combined dataset by:
1. Removing duplicate/near-duplicate images
2. Filtering low-confidence samples using a pre-trained model
3. Removing likely mislabeled samples
4. Balancing classes
5. Creating a clean, high-quality dataset

Target: Create a smaller but CLEANER dataset for higher accuracy
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm

# Emotion mapping
EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_MAP = {
    'angry': 0, 'anger': 0,
    'disgust': 1, 'disgusted': 1,
    'fear': 2, 'fearful': 2,
    'happy': 3, 'happiness': 3,
    'neutral': 4,
    'sad': 5, 'sadness': 5,
    'surprise': 6, 'surprised': 6
}


def get_image_hash(image_path, hash_size=8):
    """Get perceptual hash of image for duplicate detection"""
    try:
        img = Image.open(image_path).convert('L').resize((hash_size + 1, hash_size))
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return ''.join(['1' if b else '0' for b in diff.flatten()])
    except:
        return None


def hamming_distance(hash1, hash2):
    """Calculate hamming distance between two hashes"""
    if hash1 is None or hash2 is None:
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def collect_all_images(data_dir):
    """Collect all images from all datasets"""
    data_dir = Path(data_dir)
    all_images = []
    
    # Dataset paths to scan
    dataset_paths = [
        ('kaggle_fer/images/train', 'train'),
        ('kaggle_fer/images/validation', 'val'),
        ('emotion_detection/train', 'train'),
        ('emotion_detection/test', 'val'),
        ('mmafedb/MMAFEDB/train', 'train'),
        ('mmafedb/MMAFEDB/valid', 'val'),
        ('mmafedb/MMAFEDB/test', 'val'),
    ]
    
    for rel_path, split in dataset_paths:
        dataset_dir = data_dir / rel_path
        if not dataset_dir.exists():
            continue
            
        for emotion_dir in dataset_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            emotion_name = emotion_dir.name.lower()
            if emotion_name not in EMOTION_MAP:
                continue
                
            label = EMOTION_MAP[emotion_name]
            for img_path in emotion_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_images.append({
                        'path': str(img_path),
                        'label': label,
                        'emotion': EMOTION_NAMES[label],
                        'split': split,
                        'source': rel_path.split('/')[0]
                    })
    
    print(f"Found {len(all_images)} total images")
    return all_images


def remove_duplicates(images, threshold=5):
    """Remove duplicate and near-duplicate images"""
    print("\n🔍 Detecting duplicates...")
    
    # Calculate hashes
    hashes = {}
    for img in tqdm(images, desc="Hashing images"):
        img_hash = get_image_hash(img['path'])
        if img_hash:
            img['hash'] = img_hash
            hashes[img['path']] = img_hash
    
    # Find duplicates
    seen_hashes = {}
    duplicates = set()
    
    for img in tqdm(images, desc="Finding duplicates"):
        if 'hash' not in img:
            duplicates.add(img['path'])
            continue
            
        img_hash = img['hash']
        is_duplicate = False
        
        # Check for exact or near duplicates
        for seen_path, seen_hash in seen_hashes.items():
            if hamming_distance(img_hash, seen_hash) <= threshold:
                duplicates.add(img['path'])
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_hashes[img['path']] = img_hash
    
    # Filter out duplicates
    clean_images = [img for img in images if img['path'] not in duplicates]
    
    print(f"  Removed {len(duplicates)} duplicates")
    print(f"  Remaining: {len(clean_images)} images")
    
    return clean_images


def load_pretrained_model(device):
    """Load a pre-trained emotion model for filtering"""
    print("\n📦 Loading pre-trained model for filtering...")
    
    # Use a pre-trained EfficientNet
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)
    model = model.to(device)
    model.eval()
    
    return model


def get_model_predictions(model, images, device, batch_size=64):
    """Get model predictions for all images"""
    print("\n🔮 Getting model predictions...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    confidences = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
        batch_images = images[i:i + batch_size]
        batch_tensors = []
        
        for img_info in batch_images:
            try:
                img = Image.open(img_info['path']).convert('RGB')
                tensor = transform(img)
                batch_tensors.append(tensor)
            except:
                batch_tensors.append(torch.zeros(3, 224, 224))
        
        if not batch_tensors:
            continue
            
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            confs = probs.max(dim=1).values
        
        predictions.extend(preds.cpu().numpy())
        confidences.extend(confs.cpu().numpy())
    
    return predictions, confidences


def filter_mislabeled(images, predictions, confidences, agreement_threshold=0.5):
    """Filter out likely mislabeled samples"""
    print("\n🧹 Filtering mislabeled samples...")
    
    clean_images = []
    mislabeled_count = 0
    low_conf_count = 0
    
    for i, img in enumerate(images):
        if i >= len(predictions):
            break
            
        pred = predictions[i]
        conf = confidences[i]
        true_label = img['label']
        
        # Keep if:
        # 1. Model agrees with label (high confidence)
        # 2. Or model is uncertain (might be ambiguous expression)
        
        if pred == true_label:
            # Model agrees - keep it
            clean_images.append(img)
        elif conf < 0.4:
            # Model is uncertain - might be valid ambiguous case, keep it
            clean_images.append(img)
        elif conf > 0.8 and pred != true_label:
            # Model strongly disagrees - likely mislabeled, remove it
            mislabeled_count += 1
        else:
            # Medium confidence disagreement - be conservative, keep it
            clean_images.append(img)
    
    print(f"  Removed {mislabeled_count} likely mislabeled samples")
    print(f"  Remaining: {len(clean_images)} images")
    
    return clean_images


def balance_classes(images, max_per_class=None, min_per_class=1000):
    """Balance classes by undersampling majority classes"""
    print("\n⚖️ Balancing classes...")
    
    # Group by label
    by_label = defaultdict(list)
    for img in images:
        by_label[img['label']].append(img)
    
    # Print current distribution
    print("  Current distribution:")
    for label, imgs in sorted(by_label.items()):
        print(f"    {EMOTION_NAMES[label]:10s}: {len(imgs)}")
    
    # Determine target count
    if max_per_class is None:
        # Use median count
        counts = [len(imgs) for imgs in by_label.values()]
        max_per_class = int(np.median(counts))
    
    max_per_class = max(max_per_class, min_per_class)
    print(f"\n  Target per class: {max_per_class}")
    
    # Balance by sampling
    balanced_images = []
    for label, imgs in by_label.items():
        if len(imgs) <= max_per_class:
            balanced_images.extend(imgs)
        else:
            # Random sample
            np.random.seed(42)
            indices = np.random.choice(len(imgs), max_per_class, replace=False)
            balanced_images.extend([imgs[i] for i in indices])
    
    # Print new distribution
    by_label_new = defaultdict(list)
    for img in balanced_images:
        by_label_new[img['label']].append(img)
    
    print("\n  Balanced distribution:")
    for label, imgs in sorted(by_label_new.items()):
        print(f"    {EMOTION_NAMES[label]:10s}: {len(imgs)}")
    
    return balanced_images


def save_clean_dataset(images, output_dir):
    """Save the cleaned dataset"""
    print(f"\n💾 Saving clean dataset to {output_dir}...")
    
    output_dir = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val']:
        for emotion in EMOTION_NAMES:
            (output_dir / split / emotion).mkdir(parents=True, exist_ok=True)
    
    # Copy images
    counts = {'train': 0, 'val': 0}
    
    for img in tqdm(images, desc="Copying images"):
        split = img['split']
        emotion = img['emotion']
        src_path = Path(img['path'])
        
        # Generate unique filename
        file_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
        dst_name = f"{file_hash}_{src_path.name}"
        dst_path = output_dir / split / emotion / dst_name
        
        try:
            shutil.copy2(src_path, dst_path)
            counts[split] += 1
        except Exception as e:
            print(f"  Error copying {src_path}: {e}")
    
    print(f"\n✅ Saved {counts['train']} training images")
    print(f"✅ Saved {counts['val']} validation images")
    
    return output_dir


def main():
    print("="*60)
    print("🧹 DATASET CLEANING FOR EMOTION RECOGNITION")
    print("="*60)
    
    data_dir = 'data'
    output_dir = 'data/clean_combined'
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Step 1: Collect all images
    print("\n" + "="*60)
    print("STEP 1: Collecting images from all datasets")
    print("="*60)
    all_images = collect_all_images(data_dir)
    
    # Step 2: Remove duplicates
    print("\n" + "="*60)
    print("STEP 2: Removing duplicates")
    print("="*60)
    images = remove_duplicates(all_images, threshold=5)
    
    # Step 3: Load model and get predictions
    print("\n" + "="*60)
    print("STEP 3: Detecting mislabeled samples")
    print("="*60)
    model = load_pretrained_model(device)
    predictions, confidences = get_model_predictions(model, images, device)
    
    # Step 4: Filter mislabeled
    images = filter_mislabeled(images, predictions, confidences)
    
    # Step 5: Balance classes
    print("\n" + "="*60)
    print("STEP 4: Balancing classes")
    print("="*60)
    images = balance_classes(images, max_per_class=15000, min_per_class=3000)
    
    # Step 6: Save clean dataset
    print("\n" + "="*60)
    print("STEP 5: Saving clean dataset")
    print("="*60)
    save_clean_dataset(images, output_dir)
    
    print("\n" + "="*60)
    print("🎉 DATASET CLEANING COMPLETE!")
    print("="*60)
    print(f"\nClean dataset saved to: {output_dir}")
    print("\nNext step: Train on clean dataset with:")
    print("  python src/train_clean_dataset.py")


if __name__ == '__main__':
    main()

