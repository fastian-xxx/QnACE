"""
Q&ACE Integrated Multimodal Emotion Detection System.

This package combines facial and voice emotion detection for
comprehensive interview analysis.
"""

from pathlib import Path
import sys

# Add paths for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
FACIAL_SRC = ROOT_DIR / "interview_emotion_detection" / "src"
if str(FACIAL_SRC) not in sys.path:
    sys.path.insert(0, str(FACIAL_SRC))

# Version
__version__ = "1.0.0"

# Expose main classes
try:
    from .multimodal_detector import MultimodalEmotionDetector, MultimodalResult
    from .voice_emotion_detector import VoiceEmotionDetector
    from .interview_analyzer import MultimodalInterviewAnalyzer
    from .report_generator import generate_reports, generate_multimodal_report
except ImportError:
    # Allow importing individual modules
    pass

__all__ = [
    'MultimodalEmotionDetector',
    'MultimodalResult',
    'VoiceEmotionDetector',
    'MultimodalInterviewAnalyzer',
    'generate_reports',
    'generate_multimodal_report',
]
