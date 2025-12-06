"""
Evaluate HSEmotion model on FER2013 test set.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# Import HSEmotion
try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    print("Using HSEmotion ONNX backend")
except ImportError:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    print("Using HSEmotion PyTorch backend")

# FER2013 emotion labels (in order)
FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# HSEmotion labels (8 emotions including Contempt)
HSEMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Map HSEmotion to FER2013 indices
HSEMOTION_TO_FER = {
    0: 0,  # Anger -> angry
    2: 1,  # Disgust -> disgust
    3: 2,  # Fear -> fear
    4: 3,  # Happiness -> happy
    6: 4,  # Sadness -> sad
    7: 5,  # Surprise -> surprise
    5: 6,  # Neutral -> neutral
    1: -1, # Contempt -> skip (not in FER2013)
}


def load_fer2013(csv_path: str, split: str = 'test'):
    """Load FER2013 dataset from CSV."""
    print(f"Loading FER2013 from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by split
    if 'Usage' in df.columns:
        if split == 'test':
            df = df[df['Usage'] == 'PrivateTest']
        elif split == 'val':
            df = df[df['Usage'] == 'PublicTest']
        else:
            df = df[df['Usage'] == 'Training']
    
    images = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        # Parse pixel string
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        img = pixels.reshape(48, 48)
        
        # Convert to RGB (HSEmotion expects RGB)
        img_rgb = np.stack([img, img, img], axis=-1)
        
        images.append(img_rgb)
        labels.append(int(row['emotion']))
    
    return images, labels


def evaluate_hsemotion(images, labels, model_name='enet_b0_8_best_afew'):
    """Evaluate HSEmotion on images."""
    print(f"\nInitializing HSEmotion model: {model_name}")
    recognizer = HSEmotionRecognizer(model_name=model_name)
    
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    print("\nEvaluating...")
    for img, label in tqdm(zip(images, labels), total=len(images)):
        try:
            # Get prediction
            emotion, scores = recognizer.predict_emotions(img, logits=True)
            
            # Convert scores to probabilities
            scores = np.array(scores)
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            # Map HSEmotion prediction to FER2013 label
            # Find the max among the 7 emotions we care about (excluding Contempt)
            fer_probs = []
            for hs_idx in range(8):
                fer_idx = HSEMOTION_TO_FER.get(hs_idx, -1)
                if fer_idx >= 0:
                    fer_probs.append((fer_idx, probs[hs_idx]))
            
            # Get predicted FER label
            pred_fer_idx = max(fer_probs, key=lambda x: x[1])[0]
            
            # Check if correct
            if pred_fer_idx == label:
                correct += 1
                per_class_correct[label] += 1
            
            per_class_total[label] += 1
            total += 1
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n" + "="*60)
    print(f"HSEmotion Evaluation Results ({model_name})")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nPer-class accuracy:")
    for label_idx in range(7):
        label_name = FER_LABELS[label_idx]
        class_correct = per_class_correct[label_idx]
        class_total = per_class_total[label_idx]
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        print(f"  {label_name:10s}: {class_acc:5.2f}% ({class_correct}/{class_total})")
    
    return accuracy


def main():
    # Find FER2013 CSV
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, 'fer2013.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: FER2013 CSV not found at {csv_path}")
        print("Please ensure fer2013.csv is in the data/ directory")
        return
    
    # Load test set
    images, labels = load_fer2013(csv_path, split='test')
    print(f"Loaded {len(images)} test images")
    
    # Evaluate different models
    models = [
        'enet_b0_8_best_afew',
        # 'enet_b0_8_best_vgaf',  # Uncomment to test other models
    ]
    
    for model_name in models:
        accuracy = evaluate_hsemotion(images, labels, model_name)


if __name__ == "__main__":
    main()

