"""
Facial Emotion Detector Wrapper for Q&ACE Integration.

This module wraps the existing emotion detector for unified interface.
"""

import sys
from pathlib import Path
import torch

# Add the interview_emotion_detection src to path
INTERVIEW_SRC = Path(__file__).resolve().parents[1] / "interview_emotion_detection" / "src"
if str(INTERVIEW_SRC) not in sys.path:
    sys.path.insert(0, str(INTERVIEW_SRC))

# Import the existing emotion detector
from emotion_detector import EmotionDetector, EMOTION_LABELS

# Re-export for convenience
__all__ = ['FacialEmotionDetector', 'EMOTION_LABELS']


class FacialEmotionDetector(EmotionDetector):
    """
    Facial emotion detector wrapper for unified interface.
    
    Inherits from the existing EmotionDetector but provides
    consistent interface with VoiceEmotionDetector.
    
    Accuracy: 72.72% (EfficientNet-B2)
    Emotions: angry, disgust, fear, happy, sad, surprise, neutral
    """
    
    def __init__(self, **kwargs):
        """Initialize facial emotion detector."""
        super().__init__(**kwargs)
        self.emotions = EMOTION_LABELS
    
    def detect_emotions_from_frame(self, frame, assume_face_if_not_detected=True):
        """
        Detect emotions from a video frame.
        
        Args:
            frame: BGR image as numpy array
            assume_face_if_not_detected: If True and no face is detected,
                assume the center of the frame contains a face (common in webcam usage)
            
        Returns:
            Dict with 'emotions', 'dominant_emotion', 'confidence', 'face_detected'
        """
        results = self.detect_emotions(frame)
        
        if not results and assume_face_if_not_detected:
            # No face detected by MTCNN/Haar, but assume center region is a face
            # This is common in webcam scenarios where the user is centered
            h, w = frame.shape[:2]
            
            # Take center region (assume face is in middle 60% of frame)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.15)
            center_region = frame[margin_y:h-margin_y, margin_x:w-margin_x]
            
            # Try to analyze the center region directly
            try:
                import cv2
                from PIL import Image
                
                # Resize and preprocess
                if len(center_region.shape) == 3:
                    rgb_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
                else:
                    rgb_region = cv2.cvtColor(center_region, cv2.COLOR_GRAY2RGB)
                
                pil_img = Image.fromarray(rgb_region)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                
                emotions = {e: float(probs[i]) for i, e in enumerate(self.emotions)}
                dominant = max(emotions, key=emotions.get)
                
                return {
                    'emotions': emotions,
                    'dominant_emotion': dominant,
                    'confidence': emotions[dominant],
                    'face_detected': True,  # Assumed face
                    'face_box': (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
                }
            except Exception as e:
                print(f"Center region analysis failed: {e}")
        
        if not results:
            return {
                'emotions': {e: 0.0 for e in self.emotions},
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'face_detected': False,
                'face_box': None
            }
        
        # Get first face result
        result = results[0]
        emotions = result['emotions']
        dominant = max(emotions, key=emotions.get)
        
        return {
            'emotions': emotions,
            'dominant_emotion': dominant,
            'confidence': emotions[dominant],
            'face_detected': True,
            'face_box': result['box']
        }
