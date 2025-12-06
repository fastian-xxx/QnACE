import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from transformers import Wav2Vec2Model
import os
from datetime import datetime
import time

# ============================================
# 1. CONFIGURATION
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the correct model path - same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "QnAce_Voice-Model.pth")

# Alternative paths to check
ALTERNATIVE_PATHS = [
    os.path.join(SCRIPT_DIR, "QnAce_Voice-Model.pth"),
    r"d:\Study-Material\FYP\models\QnAce_Voice-Model\QnAce_Voice-Model.pth",
    r"d:\Study-Material\FYP\models\finetuned\best_model.pth",
    r"d:\Study-Material\FYP\models\backup\finetuned\best_model.pth",
]

SAMPLE_RATE = 16000
MAX_LENGTH = 8.0  # seconds
RECORD_DURATION = 5  # seconds for recording

EMOTIONS = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

print("="*60)
print("🎤 VOICE EMOTION TESTER")
print("="*60)
print(f"Device: {DEVICE}")

# ============================================
# 2. FIND MODEL FILE
# ============================================
def find_model():
    """Find the model file from possible locations"""
    for path in ALTERNATIVE_PATHS:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            return path
    
    # If not found, list available files
    print("\n❌ Model not found! Checking available models...")
    models_dir = r"d:\Study-Material\FYP\models"
    
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pth'):
                print(f"   Found: {os.path.join(root, file)}")
    
    return None

MODEL_PATH = find_model()
if MODEL_PATH is None:
    print("\n❌ No model file found! Please check the path.")
    print("   Expected: QnAce_Voice-Model.pth")
    exit(1)

# ============================================
# 3. MODEL ARCHITECTURE
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
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size = 768
        self.attention_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        outputs = self.wav2vec2(x).last_hidden_state
        pooled = self.attention_pool(outputs)
        logits = self.classifier(pooled)
        return logits

# ============================================
# 4. LOAD MODEL
# ============================================
print("\n📂 Loading model...")
print(f"   Path: {MODEL_PATH}")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        config = checkpoint.get('config', {})
        
        # Get emotions if available
        if 'emotions' in checkpoint:
            EMOTIONS = checkpoint['emotions']
            print(f"   Emotions: {EMOTIONS}")
        
        # Create model
        model = EmotionModel(num_classes=len(EMOTIONS), dropout=config.get('dropout', 0.3))
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Maybe the checkpoint IS the state dict
            model.load_state_dict(checkpoint)
    else:
        # Checkpoint is the state dict directly
        model = EmotionModel(num_classes=len(EMOTIONS), dropout=0.3)
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n🔧 Trying alternative loading method...")
    
    # Try loading as state dict directly
    try:
        model = EmotionModel(num_classes=6, dropout=0.3)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # If it's wrapped in a dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded with alternative method!")
    except Exception as e2:
        print(f"❌ Alternative loading also failed: {e2}")
        exit(1)

# ============================================
# 5. PREDICTION FUNCTION
# ============================================
def predict_emotion(audio_data, sr=SAMPLE_RATE):
    """Predict emotion from audio data"""
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Ensure correct length
    max_samples = int(MAX_LENGTH * SAMPLE_RATE)
    if len(audio_data) > max_samples:
        audio_data = audio_data[:max_samples]
    else:
        audio_data = np.pad(audio_data, (0, max_samples - len(audio_data)))
    
    # Convert to tensor
    waveform = torch.FloatTensor(audio_data).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    
    return {
        'emotion': EMOTIONS[pred_idx],
        'confidence': confidence,
        'all_probs': {EMOTIONS[i]: probs[i].item() for i in range(len(EMOTIONS))}
    }

# ============================================
# 6. RECORD AUDIO FUNCTION
# ============================================
def record_audio(duration=RECORD_DURATION, sr=SAMPLE_RATE):
    """Record audio from microphone"""
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   Speak now!")
    
    # Record
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    
    # Show countdown
    for i in range(duration, 0, -1):
        print(f"   {i}...", flush=True)
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    print("   Done!")
    
    return audio.flatten()

# ============================================
# 7. DISPLAY RESULTS
# ============================================
def display_results(result):
    """Display prediction results nicely"""
    emotion = result['emotion']
    confidence = result['confidence']
    emoji = EMOTION_EMOJIS.get(emotion, '❓')
    
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULT")
    print("="*50)
    
    # Main prediction
    print(f"""
┌────────────────────────────────────────────────┐
│                                                │
│   Detected Emotion: {emoji} {emotion.upper():12}           │
│   Confidence:       {confidence*100:5.1f}%                   │
│                                                │
└────────────────────────────────────────────────┘
""")
    
    # All probabilities
    print("📊 All Emotion Probabilities:")
    print("-" * 45)
    
    sorted_probs = sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        emoji_mark = EMOTION_EMOJIS.get(emo, '❓')
        marker = " 👈" if emo == emotion else ""
        print(f"   {emoji_mark} {emo:10s}: {prob*100:5.1f}% [{bar}]{marker}")
    
    # Confidence indicator
    print("\n" + "-" * 45)
    if confidence >= 0.7:
        print("   ✅ High confidence prediction")
    elif confidence >= 0.5:
        print("   🟡 Medium confidence - result may vary")
    else:
        print("   ⚠️ Low confidence - try speaking more clearly")

# ============================================
# 8. TEST FROM FILE
# ============================================
def test_from_file(file_path):
    """Test emotion from an audio file"""
    print(f"\n📁 Loading: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"   Duration: {len(audio)/sr:.2f} seconds")
    
    # Predict
    result = predict_emotion(audio, sr)
    display_results(result)
    
    return result

# ============================================
# 9. SAVE RECORDING
# ============================================
def save_recording(audio, filename=None):
    """Save recorded audio to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
    
    save_path = os.path.join(SCRIPT_DIR, filename)
    sf.write(save_path, audio, SAMPLE_RATE)
    print(f"💾 Saved recording to: {save_path}")
    return save_path

# ============================================
# 10. INTERACTIVE MENU
# ============================================
def main():
    """Main interactive loop"""
    print("\n" + "="*60)
    print("🎤 VOICE EMOTION RECOGNITION - INTERACTIVE MODE")
    print("="*60)
    print("""
Commands:
  [1] Record and test your voice (5 sec)
  [2] Test from audio file
  [3] Quick test (3 seconds)
  [4] Long test (8 seconds)
  [5] Continuous mode (keep recording)
  [q] Quit
""")
    
    while True:
        print("\n" + "-"*40)
        choice = input("Enter choice (1-5, q to quit): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            audio = record_audio(duration=5)
            result = predict_emotion(audio)
            display_results(result)
            
            save = input("\n💾 Save recording? (y/n): ").strip().lower()
            if save == 'y':
                save_recording(audio)
        
        elif choice == '2':
            file_path = input("Enter audio file path: ").strip()
            file_path = file_path.strip('"\'')
            test_from_file(file_path)
        
        elif choice == '3':
            audio = record_audio(duration=3)
            result = predict_emotion(audio)
            display_results(result)
        
        elif choice == '4':
            audio = record_audio(duration=8)
            result = predict_emotion(audio)
            display_results(result)
        
        elif choice == '5':
            print("\n🔄 CONTINUOUS MODE")
            try:
                count = 1
                while True:
                    print(f"\n--- Recording #{count} ---")
                    audio = record_audio(duration=4)
                    result = predict_emotion(audio)
                    display_results(result)
                    count += 1
                    
                    cont = input("\nPress Enter for next, 'q' to quit: ").strip().lower()
                    if cont == 'q':
                        break
            except KeyboardInterrupt:
                print("\n\n⏹️ Stopped continuous mode")
        
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q'")

# ============================================
# 11. QUICK TEST
# ============================================
def quick_test():
    """Quick single recording test"""
    print("\n🎤 Quick Voice Test")
    print("-" * 40)
    
    audio = record_audio(duration=5)
    result = predict_emotion(audio)
    display_results(result)
    
    return result

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Choose mode:")
    print("  [1] Interactive menu")
    print("  [2] Quick single test")
    print("="*60)
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == '2':
        quick_test()
    else:
        main()