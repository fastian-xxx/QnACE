# Interview Emotion Detection System

A real-time facial emotion detection system designed specifically for interview preparation and confidence scoring. This module is part of a larger interview preparation application that integrates with NLP and audio analysis modules.

## Table of Contents
- [Overview](#overview)
- [Model Evolution & Decisions](#model-evolution--decisions)
- [Architecture](#architecture)
- [Key Files for Integration](#key-files-for-integration)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Report System](#report-system)
- [Integration Guide](#integration-guide)

---

## Overview

This system analyzes facial expressions in real-time to:
- Detect 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Calculate an interview-specific **confidence score** (0-100%)
- Track performance across session segments (beginning, middle, end)
- Generate insight-first visual reports and detailed text reports
- Store session data for integration with other modules

---

## Model Evolution & Decisions

### Journey to 72.72% Accuracy

| Stage | Model | Accuracy | Why We Moved On |
|-------|-------|----------|-----------------|
| 1. Initial | `fer.FER` library | ~60% | Pre-trained, not optimized for interviews |
| 2. Fine-tuned | EfficientNet-B0 on FER2013 | 69.9% | Limited by FER2013 dataset quality |
| 3. Combined Data | EfficientNet-B0 on 200K images | 68.5% | Noisy data, needed cleaning |
| 4. Cleaned Data | EfficientNet-B0 on cleaned 180K | 70.2% | Good but overfitting |
| **5. Final** | **EfficientNet-B2 + MixUp + Label Smoothing** | **72.72%** | **Best balance of accuracy & speed** |

### Why EfficientNet-B2?

```
┌─────────────────────────────────────────────────────────────┐
│  Model Comparison                                           │
├─────────────────────────────────────────────────────────────┤
│  EfficientNet-B0:  69.9% accuracy,  ~15ms inference         │
│  EfficientNet-B2:  72.72% accuracy, ~25ms inference         │
│  ResNet-50:        68.1% accuracy,  ~20ms inference         │
│  VGG-16:           65.3% accuracy,  ~35ms inference         │
└─────────────────────────────────────────────────────────────┘
```

We chose **EfficientNet-B2** because:
1. **Best accuracy** on our cleaned dataset
2. **Real-time capable** (~40 FPS on M1 Mac)
3. **Efficient** compound scaling (depth + width + resolution)
4. **Transfer learning friendly** with ImageNet pre-training

### Why We Abandoned FER2013 Alone

FER2013 has known issues:
- **Mislabeled samples** (~10% estimated)
- **Class imbalance** (Happy: 8,989 vs Disgust: 547)
- **Low resolution** (48x48 grayscale)
- **Ambiguous expressions** in many samples

Our solution: **Combined & cleaned 180K images** from:
- FER2013 (28K)
- Face Expression Recognition Dataset (35K)
- Emotion Detection FER (35K)
- MMA Facial Expression (45K)
- Additional augmented samples

### Training Techniques That Worked

| Technique | Impact | Why |
|-----------|--------|-----|
| **MixUp Augmentation** | +1.5% | Reduces overfitting, smoother decision boundaries |
| **Label Smoothing (0.1)** | +0.8% | Handles noisy labels in FER datasets |
| **OneCycleLR Scheduler** | +1.2% | Better convergence than CosineAnnealing |
| **Focal Loss** | +0.5% | Addresses class imbalance |
| **Dropout (0.5)** | +0.7% | Regularization for generalization |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam     │────▶│  MTCNN Face      │────▶│ EfficientNet-B2 │
│   Input      │     │  Detection       │     │ Emotion Model   │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    InterviewEmotionAnalyzer                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ Emotion        │  │ Confidence     │  │ Session Data       │  │
│  │ Weights        │  │ Calculation    │  │ Storage            │  │
│  │ (interview-    │  │ (weighted      │  │ (all frames,       │  │
│  │  optimized)    │  │  scoring)      │  │  segments, spikes) │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    InterviewReportGenerator                      │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Visual Report (PNG)    │  │ Detailed Report (TXT)           │ │
│  │ - Hero score           │  │ - Full statistics               │ │
│  │ - Journey chart        │  │ - Timeline breakdown            │ │
│  │ - One action item      │  │ - Benchmarks & comparisons      │ │
│  └────────────────────────┘  └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Files for Integration

### Core Files (MUST HAVE)

```
src/
├── emotion_detector.py        # Main detector class - USE THIS
├── interview_analyzer.py      # Confidence scoring & session management
└── interview_report_generator.py  # Report generation

models/
└── best_high_accuracy_model.pth   # Trained model weights (72.72%)
```

### Integration Priority

| Priority | File | Purpose | When to Use |
|----------|------|---------|-------------|
| 🔴 **Critical** | `emotion_detector.py` | Detect emotions from frames | Every frame analysis |
| 🔴 **Critical** | `interview_analyzer.py` | Calculate confidence, store session | Real-time scoring |
| 🟡 **Important** | `interview_report_generator.py` | Generate reports | End of session |
| 🟢 **Optional** | `interview_realtime_system.py` | Full demo system | Testing/standalone |

### Files You Can Ignore

```
# Training scripts (not needed for inference)
train_*.py
clean_dataset.py
download_free_datasets.py
prepare_combined_dataset.py

# Legacy/experimental
realtime_emotion*.py
webcam_basic.py
hsemotion_detector.py  # Alternative detector, not used
```

---

## Installation

### Requirements

```bash
# Create virtual environment
python3.11 -m venv emotion_env
source emotion_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.0          # For EfficientNet-B2
facenet-pytorch      # For MTCNN face detection
opencv-python
numpy
matplotlib
Pillow
```

### Verify Installation

```bash
python tests/final_setup_test.py
```

---

## Usage

### Basic Usage

```python
from src.emotion_detector import EmotionDetector
from src.interview_analyzer import InterviewEmotionAnalyzer

# Initialize
detector = EmotionDetector()  # Auto-loads best model
analyzer = InterviewEmotionAnalyzer()

# Process a frame
import cv2
frame = cv2.imread("face.jpg")

# Detect emotions
result = detector.detect_emotions(frame)
# Returns: [{'box': (x,y,w,h), 'emotions': {'happy': 0.8, ...}}]

if result:
    emotions = result[0]['emotions']
    face_box = result[0]['box']
    
    # Get interview analysis
    analysis = analyzer.add_emotion_data(emotions, face_box)
    print(f"Confidence: {analysis['confidence_score']:.1%}")
    print(f"Level: {analysis['confidence_level']}")
```

### Real-time Processing

```python
import cv2
from src.emotion_detector import EmotionDetector
from src.interview_analyzer import InterviewEmotionAnalyzer

detector = EmotionDetector()
analyzer = InterviewEmotionAnalyzer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and analyze
    results = detector.detect_emotions(frame)
    
    if results:
        emotions = results[0]['emotions']
        face_box = results[0]['box']
        analysis = analyzer.add_emotion_data(emotions, face_box)
        
        # Display confidence
        cv2.putText(frame, f"Confidence: {analysis['confidence_score']:.0%}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Interview Practice', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Generate reports at end
from src.interview_report_generator import InterviewReportGenerator

summary = analyzer.get_session_summary()
generator = InterviewReportGenerator()
generator.generate_visual_report(summary)
generator.generate_text_report(summary)

cap.release()
```

---

## API Reference

### EmotionDetector

```python
class EmotionDetector:
    def __init__(
        self,
        model_path: str = None,      # Auto-detects best model
        use_mtcnn: bool = True,      # Use MTCNN for face detection
        device: str = None           # Auto-selects (MPS/CUDA/CPU)
    )
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect emotions in a frame.
        
        Returns:
            List of detections, each containing:
            - 'box': (x, y, w, h) face bounding box
            - 'emotions': Dict[str, float] emotion probabilities
            - 'dominant_emotion': str highest probability emotion
        """
```

### InterviewEmotionAnalyzer

```python
class InterviewEmotionAnalyzer:
    def __init__(self, window_size: int = 10)
    
    def add_emotion_data(
        self, 
        emotions: Dict[str, float],
        face_box: Tuple[int, int, int, int] = None
    ) -> Dict:
        """
        Process emotions and return analysis.
        
        Returns:
            {
                'raw_emotions': Dict,
                'smoothed_emotions': Dict,
                'confidence_score': float (0-1),
                'confidence_level': str,
                'dominant_emotion': str,
                'feedback': str,
                'color': Tuple[int, int, int]
            }
        """
    
    def get_session_summary(self) -> Dict:
        """
        Get complete session summary for reporting.
        
        Returns comprehensive dict with all metrics.
        """
    
    def save_session_data(self, path: str = "outputs/") -> str:
        """Save session data to JSON file."""
```

### InterviewReportGenerator

```python
class InterviewReportGenerator:
    def generate_visual_report(
        self,
        session_data: Dict,
        save_path: str = "outputs/"
    ) -> str:
        """Generate insight-first PNG report."""
    
    def generate_text_report(
        self,
        session_data: Dict,
        save_path: str = "outputs/"
    ) -> str:
        """Generate comprehensive TXT report."""
```

---

## Performance Metrics

### Model Accuracy (on test set)

```
┌─────────────────────────────────────────────────────────┐
│  Per-Class Accuracy (EfficientNet-B2, 72.72% overall)   │
├─────────────────────────────────────────────────────────┤
│  Happy:     90.2%  ████████████████████  (best)         │
│  Surprise:  85.8%  █████████████████                    │
│  Neutral:   74.1%  ███████████████                      │
│  Angry:     70.3%  ██████████████                       │
│  Sad:       68.5%  █████████████                        │
│  Disgust:   62.4%  ████████████                         │
│  Fear:      58.1%  ████████████  (hardest)              │
└─────────────────────────────────────────────────────────┘
```

### Inference Speed

| Device | FPS | Latency |
|--------|-----|---------|
| M1 Max (MPS) | ~40 | 25ms |
| NVIDIA RTX 3080 | ~60 | 17ms |
| CPU (Intel i7) | ~8 | 125ms |

### Confidence Score Mapping

| Score Range | Level | Interpretation |
|-------------|-------|----------------|
| 80-100% | Excellent | Outstanding interview presence |
| 65-79% | Confident | Strong professional composure |
| 50-64% | Moderate | Acceptable, room for growth |
| 35-49% | Nervous | Visible signs of anxiety |
| 20-34% | Anxious | High stress indicators |
| 0-19% | Very Anxious | Significant anxiety detected |

---

## Report System

### Visual Report (PNG)
- **Hero Score**: Large confidence percentage
- **Insight Headline**: "STRONG GROWTH", "WEAK START", etc.
- **Journey Chart**: Orange→Green gradient showing progress
- **One Action Item**: Clear next step
- **Quality Metrics**: Expressiveness, Face Detection, Stability

### Text Report (TXT)
- Session Information
- Executive Summary
- Confidence Analysis (avg, min, max, std dev)
- Segment Breakdown (beginning, middle, end)
- Timeline (10-second intervals)
- Areas for Improvement (prioritized)
- Strengths
- Benchmark Comparison
- Final Score Card

---

## Integration Guide

### With NLP Module

```python
# Your NLP module can receive the session summary
from src.interview_analyzer import InterviewEmotionAnalyzer

analyzer = InterviewEmotionAnalyzer()

# ... process video frames ...

# Get data for NLP integration
summary = analyzer.get_session_summary()

# Key fields for NLP:
nlp_data = {
    'average_confidence': summary['average_confidence'],
    'dominant_emotions': summary['emotion_percentages'],
    'stress_spikes': summary['stress_spikes_count'],
    'segment_analysis': summary['segment_analysis'],
    'timestamps': summary['time_series']['timestamps'],
    'confidence_timeline': summary['time_series']['confidence_scores']
}

# Pass to your NLP module
nlp_module.analyze_with_emotion_context(transcript, nlp_data)
```

### With Audio Module

```python
# Sync emotion data with audio timestamps
summary = analyzer.get_session_summary()

# Frame-by-frame data with timestamps
for frame in summary['all_frames']:
    timestamp = frame['timestamp']
    confidence = frame['confidence_score']
    emotion = frame['dominant_emotion']
    
    # Match with audio segment at same timestamp
    audio_segment = audio_module.get_segment_at(timestamp)
    
    # Combined analysis
    combined_score = (confidence * 0.4) + (audio_segment.confidence * 0.3) + (nlp_score * 0.3)
```

### Data Export Format

```python
# Save session for later integration
session_file = analyzer.save_session_data()

# JSON structure:
{
    "session_id": "20251128_001234",
    "session_duration": 60.0,
    "frames_analyzed": 180,
    "average_confidence": 0.65,
    "engagement_score": 0.72,
    "face_stability": 0.88,
    "face_detection_rate": 0.95,
    "stress_spikes_count": 2,
    "segment_analysis": {
        "beginning": {"avg_confidence": 0.52},
        "middle": {"avg_confidence": 0.65},
        "end": {"avg_confidence": 0.76}
    },
    "time_series": {
        "timestamps": [0.0, 0.33, 0.66, ...],
        "confidence_scores": [0.45, 0.48, 0.52, ...]
    },
    "emotion_percentages": {
        "happy": 25.5,
        "neutral": 45.2,
        ...
    }
}
```

---

## Project Structure

```
interview_emotion_detection/
├── README.md                 # This file
├── requirements.txt          # Dependencies
│
├── src/                      # Source code
│   ├── emotion_detector.py   # 🔴 CORE: Emotion detection
│   ├── interview_analyzer.py # 🔴 CORE: Confidence scoring
│   ├── interview_report_generator.py  # 🟡 Reports
│   ├── interview_realtime_system.py   # 🟢 Demo system
│   └── train_*.py            # Training scripts (not for inference)
│
├── models/                   # Trained models
│   ├── best_high_accuracy_model.pth  # 🔴 USE THIS (72.72%)
│   ├── best_emotion_model.pth        # Backup (69.9%)
│   └── ...
│
├── tests/                    # Test scripts
│   ├── test_model_accuracy.py
│   └── final_setup_test.py
│
├── outputs/                  # Generated reports
│   ├── interview_report_*.png
│   ├── interview_report_*.txt
│   └── session_data_*.json
│
└── data/                     # Training data (not needed for inference)
    └── ...
```

---

## Troubleshooting

### Common Issues

**1. MPS Out of Memory (Apple Silicon)**
```python
# Already handled in code, but if issues persist:
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
```

**2. MTCNN Not Detecting Faces**
- Ensure good lighting
- Face should be clearly visible
- Minimum face size: 60x60 pixels

**3. Low Confidence Scores**
- Model is calibrated for interview context
- Neutral expressions score 70-75% (professional)
- Only genuine smiles score 85%+

---

## Future Improvements

- [ ] Ensemble with audio sentiment for combined scoring
- [ ] Eye contact tracking integration
- [ ] Gesture recognition
- [ ] Multi-face support for panel interviews
- [ ] Model quantization for mobile deployment

---

## License

MIT License - See LICENSE file

---

## Credits

- EfficientNet architecture by Google
- MTCNN face detection by facenet-pytorch
- FER2013 dataset by Kaggle
- Training on Apple M1 Max with MPS acceleration

