"""
Integrate multiple emotion datasets into a unified training set.

Supports:
- FER2013 / FER+ (cleaned)
- AffectNet
- RAF-DB
- CK+

Usage:
    python src/integrate_datasets.py --output-dir data/unified

Expected directory structure after downloading:
    data/
    ├── fer2013.csv
    ├── fer2013new.csv (FER+ labels)
    ├── affectnet/
    │   ├── Manually_Annotated_Images/
    │   │   ├── train_set/
    │   │   └── val_set/
    │   └── Manually_Annotated_file_lists/
    │       ├── training.csv
    │       └── validation.csv
    ├── rafdb/
    │   ├── basic/
    │   │   ├── Image/
    │   │   │   ├── aligned/
    │   │   │   └── original/
    │   │   └── EmoLabel/
    │   │       └── list_patition_label.txt
    └── ck_plus/
        ├── cohn-kanade-images/
        └── Emotion/
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Unified emotion labels (7 basic emotions)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7


class DatasetIntegrator:
    """Integrates multiple emotion datasets into unified format."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        self.samples = {"train": [], "val": [], "test": []}
        self.stats = Counter()
    
    def add_fer2013(self, fer2013_path: str, ferplus_path: Optional[str] = None):
        """Add FER2013 dataset (optionally with FER+ cleaned labels)."""
        print("\n" + "="*60)
        print("Adding FER2013 dataset...")
        print("="*60)
        
        if not os.path.exists(fer2013_path):
            print(f"  ✗ FER2013 not found at {fer2013_path}")
            return
        
        # Load FER+ labels if available
        ferplus_labels = {}
        if ferplus_path and os.path.exists(ferplus_path):
            print(f"  Using FER+ cleaned labels from {ferplus_path}")
            with open(ferplus_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for idx, row in enumerate(reader):
                    if len(row) >= 12:
                        votes = [int(v) if v else 0 for v in row[2:12]]
                        # Map FER+ order to our order
                        # FER+: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
                        # Ours: angry, disgust, fear, happy, sad, surprise, neutral
                        our_votes = [
                            votes[4],  # anger -> angry
                            votes[5],  # disgust
                            votes[6],  # fear
                            votes[1],  # happiness -> happy
                            votes[3],  # sadness -> sad
                            votes[2],  # surprise
                            votes[0],  # neutral
                        ]
                        total = sum(our_votes)
                        if total > 0:
                            max_votes = max(our_votes)
                            if max_votes / total >= 0.4:  # 40% agreement threshold
                                ferplus_labels[idx] = our_votes.index(max_votes)
        
        # Load FER2013
        with open(fer2013_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                pixels = np.array([int(p) for p in row["pixels"].split()], dtype=np.uint8)
                pixels = pixels.reshape(48, 48)
                
                # Use FER+ label if available, otherwise original
                if ferplus_labels and idx in ferplus_labels:
                    label = ferplus_labels[idx]
                else:
                    label = int(row["emotion"])
                
                usage = row["Usage"]
                if usage == "Training":
                    split = "train"
                elif usage == "PublicTest":
                    split = "val"
                else:
                    split = "test"
                
                # Save image
                img_name = f"fer2013_{idx:05d}.png"
                img_path = self.output_dir / split / img_name
                Image.fromarray(pixels, mode="L").save(img_path)
                
                self.samples[split].append((str(img_path), label))
                self.stats[f"fer2013_{split}"] += 1
        
        print(f"  ✓ Added {sum(v for k, v in self.stats.items() if 'fer2013' in k)} FER2013 samples")
    
    def add_affectnet(self, affectnet_dir: str):
        """Add AffectNet dataset."""
        print("\n" + "="*60)
        print("Adding AffectNet dataset...")
        print("="*60)
        
        affectnet_path = Path(affectnet_dir)
        
        if not affectnet_path.exists():
            print(f"  ✗ AffectNet not found at {affectnet_dir}")
            print("  Download from: http://mohammadmahoor.com/affectnet/")
            return
        
        # AffectNet emotion mapping (0-7 in AffectNet)
        # AffectNet: 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger, 7=Contempt
        # Ours: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        affectnet_to_ours = {
            0: 6,  # Neutral
            1: 3,  # Happy
            2: 4,  # Sad
            3: 5,  # Surprise
            4: 2,  # Fear
            5: 1,  # Disgust
            6: 0,  # Anger
            # 7: Contempt - skip (not in our 7 basic emotions)
        }
        
        # Try different possible directory structures
        possible_structures = [
            # Structure 1: Manually_Annotated_Images with CSV
            {
                "train_csv": affectnet_path / "Manually_Annotated_file_lists" / "training.csv",
                "val_csv": affectnet_path / "Manually_Annotated_file_lists" / "validation.csv",
                "train_images": affectnet_path / "Manually_Annotated_Images" / "train_set",
                "val_images": affectnet_path / "Manually_Annotated_Images" / "val_set",
            },
            # Structure 2: Direct folders
            {
                "train_csv": affectnet_path / "training.csv",
                "val_csv": affectnet_path / "validation.csv",
                "train_images": affectnet_path / "train_set",
                "val_images": affectnet_path / "val_set",
            },
        ]
        
        found_structure = None
        for struct in possible_structures:
            if struct["train_csv"].exists() or struct["train_images"].exists():
                found_structure = struct
                break
        
        if not found_structure:
            print(f"  ✗ Could not find AffectNet data structure")
            print(f"  Expected one of:")
            print(f"    - {affectnet_path}/Manually_Annotated_file_lists/training.csv")
            print(f"    - {affectnet_path}/training.csv")
            return
        
        # Process training set
        for split, csv_key, img_key in [("train", "train_csv", "train_images"), 
                                         ("val", "val_csv", "val_images")]:
            csv_path = found_structure[csv_key]
            img_dir = found_structure[img_key]
            
            if csv_path.exists():
                print(f"  Processing AffectNet {split} from CSV...")
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    
                    for row in tqdm(reader, desc=f"  AffectNet {split}"):
                        if len(row) < 2:
                            continue
                        
                        # CSV format varies, try to find image path and label
                        img_name = row[0]
                        try:
                            label_idx = int(row[1]) if len(row) > 1 else int(row[-1])
                        except:
                            continue
                        
                        if label_idx not in affectnet_to_ours:
                            continue  # Skip contempt
                        
                        our_label = affectnet_to_ours[label_idx]
                        
                        # Find image
                        src_path = img_dir / img_name
                        if not src_path.exists():
                            # Try with different extensions
                            for ext in [".jpg", ".png", ".jpeg"]:
                                alt_path = img_dir / (Path(img_name).stem + ext)
                                if alt_path.exists():
                                    src_path = alt_path
                                    break
                        
                        if not src_path.exists():
                            continue
                        
                        # Copy image
                        dst_name = f"affectnet_{split}_{self.stats[f'affectnet_{split}']:06d}.jpg"
                        dst_path = self.output_dir / split / dst_name
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            self.samples[split].append((str(dst_path), our_label))
                            self.stats[f"affectnet_{split}"] += 1
                        except Exception as e:
                            continue
            
            elif img_dir.exists():
                # Try folder-based structure (subfolders per emotion)
                print(f"  Processing AffectNet {split} from folders...")
                for emotion_idx, our_idx in affectnet_to_ours.items():
                    emotion_dir = img_dir / str(emotion_idx)
                    if not emotion_dir.exists():
                        continue
                    
                    for img_file in tqdm(list(emotion_dir.glob("*.*")), 
                                        desc=f"  AffectNet {split}/{emotion_idx}"):
                        if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                            continue
                        
                        dst_name = f"affectnet_{split}_{self.stats[f'affectnet_{split}']:06d}.jpg"
                        dst_path = self.output_dir / split / dst_name
                        
                        try:
                            shutil.copy2(img_file, dst_path)
                            self.samples[split].append((str(dst_path), our_idx))
                            self.stats[f"affectnet_{split}"] += 1
                        except:
                            continue
        
        total = sum(v for k, v in self.stats.items() if "affectnet" in k)
        print(f"  ✓ Added {total} AffectNet samples")
    
    def add_rafdb(self, rafdb_dir: str):
        """Add RAF-DB dataset."""
        print("\n" + "="*60)
        print("Adding RAF-DB dataset...")
        print("="*60)
        
        rafdb_path = Path(rafdb_dir)
        
        if not rafdb_path.exists():
            print(f"  ✗ RAF-DB not found at {rafdb_dir}")
            print("  Download from: http://www.whdeng.cn/RAF/model1.html")
            return
        
        # RAF-DB emotion mapping
        # RAF-DB: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
        # Ours: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        rafdb_to_ours = {
            1: 5,  # Surprise
            2: 2,  # Fear
            3: 1,  # Disgust
            4: 3,  # Happiness -> happy
            5: 4,  # Sadness -> sad
            6: 0,  # Anger -> angry
            7: 6,  # Neutral
        }
        
        # Find label file
        label_file = None
        for possible in [
            rafdb_path / "basic" / "EmoLabel" / "list_patition_label.txt",
            rafdb_path / "EmoLabel" / "list_patition_label.txt",
            rafdb_path / "list_patition_label.txt",
        ]:
            if possible.exists():
                label_file = possible
                break
        
        if not label_file:
            print(f"  ✗ Could not find RAF-DB label file")
            return
        
        # Find image directory
        img_dir = None
        for possible in [
            rafdb_path / "basic" / "Image" / "aligned",
            rafdb_path / "Image" / "aligned",
            rafdb_path / "aligned",
            rafdb_path / "basic" / "Image" / "original",
        ]:
            if possible.exists():
                img_dir = possible
                break
        
        if not img_dir:
            print(f"  ✗ Could not find RAF-DB image directory")
            return
        
        print(f"  Using labels from: {label_file}")
        print(f"  Using images from: {img_dir}")
        
        # Parse label file
        with open(label_file, "r") as f:
            for line in tqdm(f.readlines(), desc="  RAF-DB"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                img_name = parts[0]
                try:
                    label = int(parts[1])
                except:
                    continue
                
                if label not in rafdb_to_ours:
                    continue
                
                our_label = rafdb_to_ours[label]
                
                # Determine split from filename (train_ or test_)
                if img_name.startswith("train_"):
                    split = "train"
                elif img_name.startswith("test_"):
                    split = "test"
                else:
                    split = "train"  # Default
                
                # Find image
                src_path = img_dir / img_name
                if not src_path.exists():
                    # Try adding _aligned suffix
                    stem = Path(img_name).stem
                    for suffix in ["_aligned.jpg", "_aligned.png", ".jpg", ".png"]:
                        alt_path = img_dir / (stem + suffix)
                        if alt_path.exists():
                            src_path = alt_path
                            break
                
                if not src_path.exists():
                    continue
                
                # Copy image
                dst_name = f"rafdb_{split}_{self.stats[f'rafdb_{split}']:05d}.jpg"
                dst_path = self.output_dir / split / dst_name
                
                try:
                    shutil.copy2(src_path, dst_path)
                    self.samples[split].append((str(dst_path), our_label))
                    self.stats[f"rafdb_{split}"] += 1
                except:
                    continue
        
        total = sum(v for k, v in self.stats.items() if "rafdb" in k)
        print(f"  ✓ Added {total} RAF-DB samples")
    
    def add_ckplus(self, ckplus_dir: str):
        """Add CK+ dataset."""
        print("\n" + "="*60)
        print("Adding CK+ dataset...")
        print("="*60)
        
        ckplus_path = Path(ckplus_dir)
        
        if not ckplus_path.exists():
            print(f"  ✗ CK+ not found at {ckplus_dir}")
            print("  Download from: http://www.jeffcohn.net/Resources/")
            return
        
        # CK+ emotion mapping (from emotion label files)
        # CK+: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
        # Ours: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        ckplus_to_ours = {
            0: 6,  # neutral
            1: 0,  # anger -> angry
            # 2: contempt - skip
            3: 1,  # disgust
            4: 2,  # fear
            5: 3,  # happy
            6: 4,  # sadness -> sad
            7: 5,  # surprise
        }
        
        # Find directories
        images_dir = None
        emotion_dir = None
        
        for img_possible in [
            ckplus_path / "cohn-kanade-images",
            ckplus_path / "CK+" / "cohn-kanade-images",
            ckplus_path / "extended-cohn-kanade-images",
        ]:
            if img_possible.exists():
                images_dir = img_possible
                break
        
        for emo_possible in [
            ckplus_path / "Emotion",
            ckplus_path / "CK+" / "Emotion",
        ]:
            if emo_possible.exists():
                emotion_dir = emo_possible
                break
        
        if not images_dir or not emotion_dir:
            print(f"  ✗ Could not find CK+ directory structure")
            return
        
        print(f"  Using images from: {images_dir}")
        print(f"  Using emotions from: {emotion_dir}")
        
        # Process each subject
        for subject_dir in sorted(emotion_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            for sequence_dir in sorted(subject_dir.iterdir()):
                if not sequence_dir.is_dir():
                    continue
                
                # Get emotion label file
                emotion_files = list(sequence_dir.glob("*.txt"))
                if not emotion_files:
                    continue
                
                # Read emotion label
                with open(emotion_files[0], "r") as f:
                    content = f.read().strip()
                    try:
                        ck_label = int(float(content))
                    except:
                        continue
                
                if ck_label not in ckplus_to_ours:
                    continue
                
                our_label = ckplus_to_ours[ck_label]
                
                # Find corresponding image (last frame of sequence)
                img_sequence_dir = images_dir / subject_dir.name / sequence_dir.name
                if not img_sequence_dir.exists():
                    continue
                
                img_files = sorted(img_sequence_dir.glob("*.png"))
                if not img_files:
                    img_files = sorted(img_sequence_dir.glob("*.jpg"))
                
                if not img_files:
                    continue
                
                # Use last frame (peak expression)
                src_path = img_files[-1]
                
                # CK+ is small, use all for training
                split = "train"
                
                dst_name = f"ckplus_{self.stats['ckplus_train']:04d}.png"
                dst_path = self.output_dir / split / dst_name
                
                try:
                    shutil.copy2(src_path, dst_path)
                    self.samples[split].append((str(dst_path), our_label))
                    self.stats["ckplus_train"] += 1
                except:
                    continue
        
        total = self.stats.get("ckplus_train", 0)
        print(f"  ✓ Added {total} CK+ samples")
    
    def save_unified_dataset(self):
        """Save unified dataset CSV files."""
        print("\n" + "="*60)
        print("Saving unified dataset...")
        print("="*60)
        
        for split in ["train", "val", "test"]:
            csv_path = self.output_dir / f"{split}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label", "emotion_name"])
                for img_path, label in self.samples[split]:
                    writer.writerow([img_path, label, EMOTION_LABELS[label]])
            
            print(f"  {split}: {len(self.samples[split])} samples -> {csv_path}")
        
        # Print statistics
        print("\n" + "="*60)
        print("UNIFIED DATASET STATISTICS")
        print("="*60)
        
        total = sum(len(v) for v in self.samples.values())
        print(f"\nTotal samples: {total}")
        
        print("\nBy source:")
        for key, count in sorted(self.stats.items()):
            print(f"  {key}: {count}")
        
        print("\nBy split:")
        for split, samples in self.samples.items():
            print(f"  {split}: {len(samples)}")
        
        print("\nBy emotion (all splits):")
        all_labels = [label for samples in self.samples.values() for _, label in samples]
        label_counts = Counter(all_labels)
        for idx, name in enumerate(EMOTION_LABELS):
            print(f"  {name:>8}: {label_counts.get(idx, 0)}")
        
        return total


def main():
    parser = argparse.ArgumentParser(description="Integrate multiple emotion datasets")
    parser.add_argument("--output-dir", type=str, default="data/unified",
                       help="Output directory for unified dataset")
    parser.add_argument("--fer2013-path", type=str, default="data/fer2013.csv")
    parser.add_argument("--ferplus-path", type=str, default="data/fer2013new.csv")
    parser.add_argument("--affectnet-dir", type=str, default="data/affectnet")
    parser.add_argument("--rafdb-dir", type=str, default="data/rafdb")
    parser.add_argument("--ckplus-dir", type=str, default="data/ck_plus")
    parser.add_argument("--skip-fer2013", action="store_true")
    parser.add_argument("--skip-affectnet", action="store_true")
    parser.add_argument("--skip-rafdb", action="store_true")
    parser.add_argument("--skip-ckplus", action="store_true")
    args = parser.parse_args()
    
    integrator = DatasetIntegrator(args.output_dir)
    
    # Add datasets
    if not args.skip_fer2013:
        integrator.add_fer2013(args.fer2013_path, args.ferplus_path)
    
    if not args.skip_affectnet:
        integrator.add_affectnet(args.affectnet_dir)
    
    if not args.skip_rafdb:
        integrator.add_rafdb(args.rafdb_dir)
    
    if not args.skip_ckplus:
        integrator.add_ckplus(args.ckplus_dir)
    
    # Save unified dataset
    total = integrator.save_unified_dataset()
    
    if total > 0:
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nUnified dataset saved to: {args.output_dir}/")
        print("\nTo train on this dataset, run:")
        print(f"  python src/train_unified.py --data-dir {args.output_dir} --epochs 50")
    else:
        print("\n⚠️  No samples were added. Please check dataset paths.")


if __name__ == "__main__":
    main()

