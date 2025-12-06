"""
Voice Emotion Detector Module for Q&ACE Integration.

This module provides voice/audio emotion detection using a fine-tuned
Wav2Vec2 model with attention pooling.

Accuracy: 73.37% on test set
Emotions: anger, fear, happy, neutral, sad, surprise
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import librosa

# Try to import transformers for Wav2Vec2
try:
    from transformers import Wav2Vec2Model
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False
    print("Warning: transformers not installed. Voice detection unavailable.")

# Constants
VOICE_EMOTIONS = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SAMPLE_RATE = 16000
MAX_LENGTH = 8.0  # seconds

EMOTION_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class VoiceEmotionModel(nn.Module):
    """Voice emotion classification model using Wav2Vec2."""
    
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        
        if not WAV2VEC_AVAILABLE:
            raise ImportError("transformers required for voice emotion detection")
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(x).last_hidden_state
        pooled = self.attention_pool(outputs)
        logits = self.classifier(pooled)
        return logits


class VoiceEmotionDetector:
    """
    Voice emotion detector using fine-tuned Wav2Vec2 model.
    
    Provides similar interface to EmotionDetector for easy integration.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize voice emotion detector.
        
        Args:
            model_path: Path to model checkpoint. Auto-detects if None.
            device: Device for inference ('cuda', 'mps', 'cpu', or None for auto)
        """
        if not WAV2VEC_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers")
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"VoiceEmotionDetector using device: {self.device}")
        
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        self.model_path = Path(model_path)
        self.emotions = VOICE_EMOTIONS
        
        # Load model
        self._load_model()
        
    def _find_model(self) -> Path:
        """Auto-detect model path."""
        # Check relative to this file
        repo_root = Path(__file__).resolve().parents[1]
        
        possible_paths = [
            repo_root / "QnAce_Voice-Model" / "QnAce_Voice-Model.pth",
            repo_root / "models" / "QnAce_Voice-Model.pth",
            Path.home() / "models" / "QnAce_Voice-Model.pth",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return first path (will show warning if not found)
        return possible_paths[0]
    
    def _load_model(self):
        """Load model weights from checkpoint."""
        print(f"Loading voice model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Voice model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint
        if isinstance(checkpoint, dict):
            config = checkpoint.get('config', {})
            if 'emotions' in checkpoint:
                self.emotions = checkpoint['emotions']
            
            dropout = config.get('dropout', 0.3)
        else:
            dropout = 0.3
        
        # Build model
        self.model = VoiceEmotionModel(
            num_classes=len(self.emotions),
            dropout=dropout
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Voice model loaded! Emotions: {self.emotions}")
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ) -> torch.Tensor:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of input audio
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        # Ensure correct length
        max_samples = int(MAX_LENGTH * SAMPLE_RATE)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Convert to tensor
        waveform = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        return waveform
    
    def detect_emotions(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ) -> Dict:
        """
        Detect emotions from audio.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dict with 'emotions' (probabilities), 'dominant_emotion', 'confidence'
        """
        # Preprocess
        waveform = self.preprocess_audio(audio, sample_rate)
        
        # Predict
        with torch.no_grad():
            logits = self.model(waveform)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        # Build result
        emotions = {
            self.emotions[i]: float(probs[i].cpu())
            for i in range(len(self.emotions))
        }
        
        return {
            'emotions': emotions,
            'dominant_emotion': self.emotions[pred_idx],
            'confidence': confidence
        }
    
    def detect_emotions_from_file(self, file_path: str) -> Dict:
        """
        Detect emotions from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with emotion predictions
        """
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return self.detect_emotions(audio, sr)


# Convenience function for quick testing
def detect_voice_emotion(audio_path: str) -> Dict:
    """Quick function to detect emotion from audio file."""
    detector = VoiceEmotionDetector()
    return detector.detect_emotions_from_file(audio_path)
