"""
Download and prepare FREE emotion datasets (no academic license required).

Available datasets:
1. FER2013 - You already have this ✓
2. FER+ labels - Already downloaded ✓
3. JAFFE - Japanese Female Facial Expression (213 images, very clean)
4. CK+ subset on Kaggle - Available on Kaggle
5. KDEF - Karolinska Directed Emotional Faces
6. Natural Human Face Images (Kaggle)
7. Face Expression Recognition Dataset (Kaggle)

Usage:
    python src/download_free_datasets.py --output-dir data/free_datasets
"""

import argparse
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
import urllib.request
import csv

# URLs for free datasets
DATASET_URLS = {
    # These are example URLs - some may need Kaggle CLI
    "fer_plus": "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv",
}

KAGGLE_DATASETS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FREE DATASETS ON KAGGLE (No License Required!)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  1. Face Expression Recognition Dataset (~35K images)                         ║
║     URL: https://www.kaggle.com/datasets/jonathanoheix/face-expression-       ║
║          recognition-dataset                                                   ║
║     Download: Click "Download" button on Kaggle                               ║
║     Place in: data/kaggle_fer/                                                ║
║                                                                                ║
║  2. Natural Human Face Images for Emotion Recognition                          ║
║     URL: https://www.kaggle.com/datasets/sudarshanvaidya/random-images-       ║
║          for-face-emotion-recognition                                          ║
║     Download: Click "Download" button                                          ║
║     Place in: data/natural_faces/                                             ║
║                                                                                ║
║  3. Facial Expression Dataset (Another FER variant)                           ║
║     URL: https://www.kaggle.com/datasets/astraszab/facial-expression-         ║
║          dataset-image-folders-fer2013                                        ║
║     Download: Click "Download" button                                          ║
║     Place in: data/fer_images/                                                ║
║                                                                                ║
║  4. Emotion Detection (Real faces)                                            ║
║     URL: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer     ║
║     Download: Click "Download" button                                          ║
║     Place in: data/emotion_detection/                                         ║
║                                                                                ║
║  5. MMAFEDB - Multi-Media Affective Faces                                      ║
║     URL: https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression     ║
║     Download: Click "Download" button                                          ║
║     Place in: data/mmafedb/                                                   ║
║                                                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  HOW TO DOWNLOAD FROM KAGGLE:                                                  ║
║  1. Create a free Kaggle account at kaggle.com                                ║
║  2. Go to the dataset URL                                                     ║
║  3. Click the "Download" button (top right)                                   ║
║  4. Extract the ZIP file to the specified folder                              ║
║                                                                                ║
║  OR use Kaggle CLI:                                                           ║
║  pip install kaggle                                                           ║
║  kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def print_kaggle_instructions():
    """Print instructions for downloading Kaggle datasets."""
    print(KAGGLE_DATASETS)


def setup_kaggle_cli():
    """Check if Kaggle CLI is set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         KAGGLE CLI SETUP REQUIRED                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  To use Kaggle CLI for automatic downloads:                                   ║
║                                                                                ║
║  1. Go to https://www.kaggle.com/settings                                     ║
║  2. Scroll to "API" section                                                   ║
║  3. Click "Create New Token" - this downloads kaggle.json                     ║
║  4. Move kaggle.json to ~/.kaggle/kaggle.json                                 ║
║  5. Run: chmod 600 ~/.kaggle/kaggle.json                                      ║
║                                                                                ║
║  Then run this script again!                                                  ║
║                                                                                ║
║  OR: Just download manually from Kaggle website (easier!)                     ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        return False
    return True


def download_with_kaggle_cli(dataset_id: str, output_dir: str):
    """Download dataset using Kaggle CLI."""
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
        return True
    except Exception as e:
        print(f"  Kaggle CLI download failed: {e}")
        return False


def process_kaggle_fer(data_dir: str, output_dir: str):
    """
    Process Kaggle Face Expression Recognition Dataset.
    Expected structure:
        data_dir/
        ├── train/
        │   ├── angry/
        │   ├── disgust/
        │   ├── fear/
        │   ├── happy/
        │   ├── sad/
        │   ├── surprise/
        │   └── neutral/
        └── validation/
            └── ... (same structure)
    """
    print("\nProcessing Kaggle FER dataset...")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"  ✗ Dataset not found at {data_dir}")
        print(f"    Download from Kaggle and extract to {data_dir}")
        return 0
    
    samples = {"train": [], "val": [], "test": []}
    
    # Map folder names to our label indices
    folder_to_label = {
        "angry": 0, "anger": 0,
        "disgust": 1, "disgusted": 1,
        "fear": 2, "fearful": 2,
        "happy": 3, "happiness": 3,
        "sad": 4, "sadness": 4,
        "surprise": 5, "surprised": 5,
        "neutral": 6,
    }
    
    total_count = 0
    
    for split_folder in ["train", "training", "Training"]:
        split_path = data_path / split_folder
        if not split_path.exists():
            continue
        
        for emotion_folder in split_path.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name.lower()
            if emotion_name not in folder_to_label:
                continue
            
            label = folder_to_label[emotion_name]
            
            for img_file in emotion_folder.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Copy to output
                    dst_name = f"kaggle_fer_train_{total_count:06d}{img_file.suffix}"
                    dst_path = output_path / "train" / dst_name
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, dst_path)
                    samples["train"].append((str(dst_path), label))
                    total_count += 1
    
    for split_folder in ["validation", "val", "test", "Testing"]:
        split_path = data_path / split_folder
        if not split_path.exists():
            continue
        
        split_name = "val" if "val" in split_folder.lower() else "test"
        
        for emotion_folder in split_path.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name.lower()
            if emotion_name not in folder_to_label:
                continue
            
            label = folder_to_label[emotion_name]
            
            for img_file in emotion_folder.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    dst_name = f"kaggle_fer_{split_name}_{total_count:06d}{img_file.suffix}"
                    dst_path = output_path / split_name / dst_name
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, dst_path)
                    samples[split_name].append((str(dst_path), label))
                    total_count += 1
    
    print(f"  ✓ Processed {total_count} images from Kaggle FER")
    return samples


def process_natural_faces(data_dir: str, output_dir: str):
    """Process Natural Human Face Images dataset."""
    print("\nProcessing Natural Faces dataset...")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"  ✗ Dataset not found at {data_dir}")
        return {}
    
    samples = {"train": [], "val": [], "test": []}
    
    folder_to_label = {
        "angry": 0, "anger": 0,
        "disgust": 1, "disgusted": 1,
        "fear": 2, "fearful": 2,
        "happy": 3, "happiness": 3,
        "sad": 4, "sadness": 4,
        "surprise": 5, "surprised": 5,
        "neutral": 6,
    }
    
    total_count = 0
    
    # Try different possible structures
    for root_folder in [data_path, data_path / "images", data_path / "train"]:
        if not root_folder.exists():
            continue
        
        for emotion_folder in root_folder.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name.lower()
            if emotion_name not in folder_to_label:
                continue
            
            label = folder_to_label[emotion_name]
            
            for img_file in emotion_folder.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    dst_name = f"natural_{total_count:06d}{img_file.suffix}"
                    dst_path = output_path / "train" / dst_name
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, dst_path)
                    samples["train"].append((str(dst_path), label))
                    total_count += 1
    
    print(f"  ✓ Processed {total_count} images from Natural Faces")
    return samples


def create_unified_csvs(all_samples: dict, output_dir: str):
    """Create unified CSV files from all samples."""
    output_path = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        csv_path = output_path / f"{split}.csv"
        samples = all_samples.get(split, [])
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "emotion_name"])
            for img_path, label in samples:
                writer.writerow([img_path, label, EMOTION_LABELS[label]])
        
        print(f"  {split}.csv: {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare free emotion datasets")
    parser.add_argument("--output-dir", type=str, default="data/free_combined",
                       help="Output directory for combined dataset")
    parser.add_argument("--kaggle-fer-dir", type=str, default="data/kaggle_fer",
                       help="Directory containing Kaggle FER dataset")
    parser.add_argument("--natural-faces-dir", type=str, default="data/natural_faces",
                       help="Directory containing Natural Faces dataset")
    parser.add_argument("--fer2013-path", type=str, default="data/fer2013.csv",
                       help="Path to FER2013 CSV")
    parser.add_argument("--ferplus-path", type=str, default="data/fer2013new.csv",
                       help="Path to FER+ labels")
    parser.add_argument("--show-instructions", action="store_true",
                       help="Show download instructions")
    parser.add_argument("--try-kaggle-cli", action="store_true",
                       help="Try to download using Kaggle CLI")
    args = parser.parse_args()
    
    print("="*70)
    print("FREE EMOTION DATASET DOWNLOADER & INTEGRATOR")
    print("="*70)
    
    if args.show_instructions:
        print_kaggle_instructions()
        return
    
    # Show instructions first
    print_kaggle_instructions()
    
    # Try Kaggle CLI if requested
    if args.try_kaggle_cli:
        if setup_kaggle_cli():
            print("\nAttempting Kaggle CLI downloads...")
            
            # Download Face Expression Recognition Dataset
            print("\nDownloading jonathanoheix/face-expression-recognition-dataset...")
            success = download_with_kaggle_cli(
                "jonathanoheix/face-expression-recognition-dataset",
                args.kaggle_fer_dir
            )
            if success:
                print("  ✓ Downloaded successfully!")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(exist_ok=True)
    
    all_samples = {"train": [], "val": [], "test": []}
    
    # Process FER2013 with FER+ labels (you already have this)
    print("\n" + "="*70)
    print("PROCESSING AVAILABLE DATASETS")
    print("="*70)
    
    if os.path.exists(args.fer2013_path):
        print("\nProcessing FER2013 with FER+ labels...")
        
        # Load FER+ labels
        ferplus_labels = {}
        if os.path.exists(args.ferplus_path):
            with open(args.ferplus_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for idx, row in enumerate(reader):
                    if len(row) >= 12:
                        votes = [int(v) if v else 0 for v in row[2:12]]
                        our_votes = [votes[4], votes[5], votes[6], votes[1], votes[3], votes[2], votes[0]]
                        total = sum(our_votes)
                        if total > 0:
                            max_votes = max(our_votes)
                            if max_votes / total >= 0.4:
                                ferplus_labels[idx] = our_votes.index(max_votes)
        
        # Process FER2013
        import numpy as np
        from PIL import Image
        
        with open(args.fer2013_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if ferplus_labels and idx not in ferplus_labels:
                    continue
                
                pixels = np.array([int(p) for p in row["pixels"].split()], dtype=np.uint8)
                pixels = pixels.reshape(48, 48)
                
                label = ferplus_labels.get(idx, int(row["emotion"]))
                
                usage = row["Usage"]
                if usage == "Training":
                    split = "train"
                elif usage == "PublicTest":
                    split = "val"
                else:
                    split = "test"
                
                img_name = f"fer2013_{idx:05d}.png"
                img_path = output_path / split / img_name
                Image.fromarray(pixels, mode="L").save(img_path)
                
                all_samples[split].append((str(img_path), label))
        
        print(f"  ✓ Processed {sum(len(v) for v in all_samples.values())} FER2013 images")
    
    # Process Kaggle FER if available
    kaggle_samples = process_kaggle_fer(args.kaggle_fer_dir, args.output_dir)
    if kaggle_samples:
        for split in ["train", "val", "test"]:
            all_samples[split].extend(kaggle_samples.get(split, []))
    
    # Process Natural Faces if available
    natural_samples = process_natural_faces(args.natural_faces_dir, args.output_dir)
    if natural_samples:
        for split in ["train", "val", "test"]:
            all_samples[split].extend(natural_samples.get(split, []))
    
    # Create unified CSVs
    print("\n" + "="*70)
    print("CREATING UNIFIED DATASET")
    print("="*70)
    create_unified_csvs(all_samples, args.output_dir)
    
    # Print summary
    total = sum(len(v) for v in all_samples.values())
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal samples: {total}")
    print(f"\nBy split:")
    for split, samples in all_samples.items():
        print(f"  {split}: {len(samples)}")
    
    print(f"\nDataset saved to: {args.output_dir}/")
    
    if total > 35000:
        print("\n✓ Dataset is larger than FER2013 alone!")
        print("\nTo train on this dataset:")
        print(f"  python src/train_unified.py --data-dir {args.output_dir} --epochs 50")
    else:
        print("\n⚠️  To increase dataset size, download more datasets from Kaggle:")
        print("  python src/download_free_datasets.py --show-instructions")


if __name__ == "__main__":
    main()

