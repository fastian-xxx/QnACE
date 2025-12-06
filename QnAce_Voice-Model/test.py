import torch
import torch.nn as nn
import numpy as np
import os
import json
import librosa
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Wav2Vec2Model

# ============================================
# 1. CONFIGURATION
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"d:\Study-Material\FYP\models\ep14\ep14.pth"
VAL_DIR = r"d:\Study-Material\FYP\dataset\split\val"
TEST_DIR = r"d:\Study-Material\FYP\dataset\split\test"
SAMPLE_RATE = 16000
MAX_LENGTH = 8.0

print("="*60)
print("🧪 TESTING EP14 MODEL")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# ============================================
# 2. LOAD CHECKPOINT FIRST
# ============================================
print("\n📂 Loading checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Get info from checkpoint
EMOTIONS = checkpoint.get('emotions', ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
config = checkpoint.get('config', {})
print(f"🎯 Emotions: {EMOTIONS}")
print(f"📋 Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"📊 Val Acc: {checkpoint.get('val_acc', 'unknown'):.4f}")

# ============================================
# 3. DEFINE MODEL ARCHITECTURE (MATCHING TRAINING)
# ============================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class EmotionModel(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        
        # Load Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        hidden_size = 768
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_size)
        
        # Classifier - Using LayerNorm (no running stats needed)
        # Indices: 0,1 = Linear+LayerNorm, 4,5 = Linear+LayerNorm, 8,9 = Linear+LayerNorm, 12 = Final Linear
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),      # 0
            nn.LayerNorm(512),                # 1
            nn.ReLU(),                        # 2
            nn.Dropout(dropout),              # 3
            nn.Linear(512, 256),              # 4
            nn.LayerNorm(256),                # 5
            nn.ReLU(),                        # 6
            nn.Dropout(dropout),              # 7
            nn.Linear(256, 128),              # 8
            nn.LayerNorm(128),                # 9
            nn.ReLU(),                        # 10
            nn.Dropout(dropout),              # 11
            nn.Linear(128, num_classes)       # 12
        )
    
    def forward(self, x):
        outputs = self.wav2vec2(x).last_hidden_state
        pooled = self.attention_pool(outputs)
        logits = self.classifier(pooled)
        return logits

# ============================================
# 4. CREATE MODEL AND LOAD WEIGHTS
# ============================================
print("\n🧠 Creating model...")
model = EmotionModel(num_classes=len(EMOTIONS), dropout=config.get('dropout', 0.3))

print("📥 Loading weights...")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"✅ Model loaded successfully!")

# ============================================
# 5. LOAD AUDIO FUNCTION
# ============================================
def load_audio(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        max_samples = int(MAX_LENGTH * SAMPLE_RATE)
        
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        else:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)))
        
        return torch.FloatTensor(waveform)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# ============================================
# 6. EVALUATION FUNCTION
# ============================================
def evaluate_folder(data_dir, name="Data"):
    print(f"\n{'='*60}")
    print(f"🧪 EVALUATING: {name}")
    print(f"{'='*60}")
    
    all_preds = []
    all_labels = []
    
    # Load all files
    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"⚠️ Missing folder: {emotion_dir}")
            continue
        
        files = list(Path(emotion_dir).glob("*.wav"))
        print(f"   {emotion}: {len(files)} files")
        
        for file_path in tqdm(files, desc=f"   {emotion}", leave=False):
            waveform = load_audio(file_path)
            if waveform is None:
                continue
            
            # Predict
            with torch.no_grad():
                waveform = waveform.unsqueeze(0).to(DEVICE)
                output = model(waveform)
                pred = torch.argmax(output, dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(emotion_idx)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    
    print(f"\n📊 {name} Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\n📈 Per-Class Accuracy:")
    print("-" * 45)
    per_class = {}
    for i, emotion in enumerate(EMOTIONS):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            per_class[emotion] = acc
            bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
            print(f"   {emotion:10s}: {acc:5.1f}% [{bar}]")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'{name} Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save confusion matrix
    save_path = f'd:/Study-Material/FYP/models/ep14/{name.lower()}_confusion_matrix.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved: {save_path}")
    
    return {
        'accuracy': accuracy,
        'per_class': per_class
    }

# ============================================
# 7. RUN EVALUATION
# ============================================
val_results = evaluate_folder(VAL_DIR, "VALIDATION")
test_results = evaluate_folder(TEST_DIR, "TEST")

# ============================================
# 8. SUMMARY
# ============================================
print("\n" + "="*60)
print("🎯 FINAL SUMMARY - EP14 MODEL")
print("="*60)

print(f"""
┌─────────────────────────────────────────┐
│  VALIDATION: {val_results['accuracy']:5.2f}%                      │
│  TEST:       {test_results['accuracy']:5.2f}%                      │
└─────────────────────────────────────────┘
""")

print("Per-Class Comparison:")
print("-" * 50)
print(f"{'Emotion':<12} {'Val Acc':>10} {'Test Acc':>10} {'Diff':>10}")
print("-" * 50)
for emotion in EMOTIONS:
    val_acc = val_results['per_class'].get(emotion, 0)
    test_acc = test_results['per_class'].get(emotion, 0)
    diff = test_acc - val_acc
    print(f"{emotion:<12} {val_acc:>9.1f}% {test_acc:>9.1f}% {diff:>+9.1f}%")

# ============================================
# 9. SAVE RESULTS
# ============================================
results = {
    'model': 'ep14.pth',
    'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
    'checkpoint_val_acc': float(checkpoint.get('val_acc', 0)),
    'validation': {
        'accuracy': val_results['accuracy'],
        'per_class': val_results['per_class']
    },
    'test': {
        'accuracy': test_results['accuracy'],
        'per_class': test_results['per_class']
    }
}

results_path = r'd:\Study-Material\FYP\models\ep14\ep14_test_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved: {results_path}")
print("\n🎉 EVALUATION COMPLETE!")