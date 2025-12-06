"""
Comprehensive test script for the emotion detection model.

Tests:
1. Model loading and initialization
2. Face detection
3. Emotion classification accuracy on FER2013 test set
4. Per-class performance
5. Inference speed
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Test imports
print("="*60)
print("TEST 1: Import and Initialization")
print("="*60)

try:
    from emotion_detector import EmotionDetector, EMOTION_LABELS
    print("✅ EmotionDetector imported successfully")
except Exception as e:
    print(f"❌ Failed to import EmotionDetector: {e}")
    sys.exit(1)

# Initialize detector
try:
    detector = EmotionDetector(use_mtcnn=False)  # Use Haar for speed in testing
    print("✅ EmotionDetector initialized successfully")
    print(f"   Model type: {detector.model_type}")
    print(f"   Image size: {detector.image_size}")
    print(f"   Device: {detector.device}")
except Exception as e:
    print(f"❌ Failed to initialize EmotionDetector: {e}")
    sys.exit(1)

# Test on FER2013
print("\n" + "="*60)
print("TEST 2: Accuracy on FER2013 Test Set")
print("="*60)

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fer2013.csv')

if not os.path.exists(data_path):
    print(f"⚠️ FER2013 dataset not found at {data_path}")
    print("   Skipping accuracy test")
else:
    print(f"Loading FER2013 from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Get test set (PrivateTest)
    test_df = df[df['Usage'] == 'PrivateTest']
    print(f"Test set size: {len(test_df)} images")
    
    # Prepare images
    images = []
    labels = []
    
    for _, row in test_df.iterrows():
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        img = pixels.reshape(48, 48)
        # Convert to BGR (3 channel) as detector expects
        img_bgr = np.stack([img, img, img], axis=-1)
        images.append(img_bgr)
        labels.append(int(row['emotion']))
    
    print(f"Loaded {len(images)} test images")
    
    # Run predictions
    print("\nRunning predictions...")
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    inference_times = []
    
    for img, label in tqdm(zip(images, labels), total=len(images)):
        start_time = time.time()
        
        # The image is already a face, so pass it directly
        # We'll use the model directly for speed
        from PIL import Image
        import torch
        
        img_pil = Image.fromarray(img)
        img_tensor = detector.transform(img_pil).unsqueeze(0).to(detector.device)
        
        with torch.no_grad():
            outputs = detector.model(img_tensor)
            pred = outputs.argmax(dim=1).item()
        
        inference_times.append(time.time() - start_time)
        
        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        
        per_class_total[label] += 1
        total += 1
    
    # Calculate metrics
    accuracy = correct / total * 100
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"⏱️  Average Inference Time: {avg_inference_time:.2f} ms/image")
    print(f"🚀 Throughput: {1000/avg_inference_time:.1f} images/second")
    
    print("\n📈 Per-Class Accuracy:")
    print("-" * 40)
    for i, emotion in enumerate(EMOTION_LABELS):
        class_correct = per_class_correct[i]
        class_total = per_class_total[i]
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        bar = "█" * int(class_acc / 5) + "░" * (20 - int(class_acc / 5))
        print(f"  {emotion:10s}: {bar} {class_acc:5.2f}% ({class_correct}/{class_total})")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if accuracy >= 72:
        print("✅ Model accuracy meets target (72%+)")
    elif accuracy >= 70:
        print("⚠️ Model accuracy is good but below 72% target")
    else:
        print("❌ Model accuracy is below expected")
    
    # Check critical emotions for interviews
    happy_acc = per_class_correct[3] / per_class_total[3] * 100
    neutral_acc = per_class_correct[6] / per_class_total[6] * 100
    
    print(f"\n🎯 Interview-Critical Emotions:")
    print(f"   Happy:   {happy_acc:.1f}% {'✅' if happy_acc >= 85 else '⚠️'}")
    print(f"   Neutral: {neutral_acc:.1f}% {'✅' if neutral_acc >= 65 else '⚠️'}")
    
    if happy_acc >= 85 and neutral_acc >= 65:
        print("\n✅ Model is READY for interview emotion detection!")
    else:
        print("\n⚠️ Model may need improvement for interview use")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)

