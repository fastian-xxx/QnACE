"""
Custom emotion detector using the fine-tuned EfficientNet model.

This module provides a drop-in replacement for FER that uses our fine-tuned model
with significantly better accuracy on fear, disgust, and neutral emotions.

Supports both EfficientNet-B0 (69.9% accuracy) and EfficientNet-B2 (72.72% accuracy).
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Try to import timm for EfficientNet-B2
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    from torchvision import models

# Try to import MTCNN for face detection
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# Emotion labels in FER2013 order
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def build_model_b0(num_classes: int = 7) -> nn.Module:
    """Build EfficientNet-B0 architecture (old model)."""
    from torchvision import models
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


def build_model_b2(num_classes: int = 7, dropout: float = 0.4) -> nn.Module:
    """Build EfficientNet-B2 architecture (new high-accuracy model)."""
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for EfficientNet-B2. Install with: pip install timm")
    
    model = timm.create_model('efficientnet_b2.ra_in1k', pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.75),
        nn.Linear(512, num_classes)
    )
    return model


def build_model(num_classes: int = 7, model_type: str = 'b2') -> nn.Module:
    """Build model based on type."""
    if model_type == 'b2':
        return build_model_b2(num_classes)
    else:
        return build_model_b0(num_classes)


class EmotionDetector:
    """
    Custom emotion detector using fine-tuned EfficientNet models.
    
    Supports:
    - EfficientNet-B2 (72.72% accuracy) - default, best_high_accuracy_model.pth
    - EfficientNet-B0 (69.9% accuracy) - fallback, best_emotion_model.pth
    
    This class provides a similar interface to FER for easy integration
    with existing code.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mtcnn: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the emotion detector.
        
        Args:
            model_path: Path to the fine-tuned model checkpoint.
                       If None, auto-detects best available model.
            use_mtcnn: Whether to use MTCNN for face detection
            device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto)
        """
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
        
        print(f"EmotionDetector using device: {self.device}")
        
        # Auto-detect best model
        repo_root = Path(__file__).resolve().parents[1]
        
        if model_path is None:
            # Try high-accuracy B2 model first, then fall back to B0
            b2_path = repo_root / "models" / "best_high_accuracy_model.pth"
            b0_path = repo_root / "models" / "best_emotion_model.pth"
            
            if os.path.exists(b2_path):
                model_path = b2_path
                self.model_type = 'b2'
                self.image_size = 260
            elif os.path.exists(b0_path):
                model_path = b0_path
                self.model_type = 'b0'
                self.image_size = 224
            else:
                model_path = b2_path  # Will show warning
                self.model_type = 'b2'
                self.image_size = 260
        else:
            # Detect model type from path
            model_path = Path(model_path)
            if 'high_accuracy' in str(model_path) or 'b2' in str(model_path):
                self.model_type = 'b2'
                self.image_size = 260
            else:
                self.model_type = 'b0'
                self.image_size = 224
        
        # Build model
        self.model = build_model(num_classes=len(EMOTION_LABELS), model_type=self.model_type)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            accuracy = checkpoint.get('val_acc', 'unknown')
            print(f"Loaded {self.model_type.upper()} model from {model_path}")
            print(f"Model accuracy: {accuracy}%")
        else:
            print(f"WARNING: Model not found at {model_path}, using random weights!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup face detector
        # IMPORTANT: MTCNN has issues with MPS device, always use CPU for face detection
        self.use_mtcnn = use_mtcnn and MTCNN_AVAILABLE
        if self.use_mtcnn:
            # Force CPU for MTCNN - MPS has interpolation issues
            self.face_detector = MTCNN(
                keep_all=True,
                device="cpu"  # Always CPU for MTCNN due to MPS issues
            )
            print("Using MTCNN for face detection (CPU)")
        else:
            # Fallback to Haar cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            print("Using Haar cascade for face detection")
        
        # Image transforms (use correct size for model)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        # MTCNN expects RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, probs = self.face_detector.detect(image_rgb)
            
            faces = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Filter by probability threshold
                    if probs is not None and probs[i] is not None and probs[i] > 0.5:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        w, h = x2 - x1, y2 - y1
                        if w > 20 and h > 20:  # Minimum face size
                            faces.append((x1, y1, w, h))
            
            return faces
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return []
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try with different parameters for better detection
        for scale, neighbors in [(1.1, 3), (1.05, 3), (1.2, 5)]:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(30, 30),
            )
            if len(faces) > 0:
                return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        
        return []
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image with fallback strategy.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of (x, y, w, h) tuples for each detected face
        """
        faces = []
        
        # Try MTCNN first if available
        if self.use_mtcnn:
            faces = self._detect_faces_mtcnn(image)
        
        # Fallback to Haar cascade if MTCNN fails
        if not faces:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            haar_detector = cv2.CascadeClassifier(cascade_path)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            for scale, neighbors in [(1.1, 3), (1.05, 2), (1.2, 4)]:
                detected = haar_detector.detectMultiScale(
                    gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(30, 30)
                )
                if len(detected) > 0:
                    faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]
                    break
        
        return faces
    
    def _preprocess_face(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> torch.Tensor:
        """Extract and preprocess a face region."""
        x, y, w, h = box
        
        # Add padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        face = image[y1:y2, x1:x2]
        
        # Convert BGR to RGB
        if len(face.shape) == 3:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL and apply transforms
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil)
        
        return face_tensor
    
    def detect_emotions(
        self,
        image: np.ndarray,
        face_rectangles: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict]:
        """
        Detect emotions in faces within an image.
        
        Args:
            image: BGR image as numpy array
            face_rectangles: Optional list of (x, y, w, h) face boxes.
                           If None, faces will be detected automatically.
        
        Returns:
            List of dicts with 'box' and 'emotions' keys for each face.
            'emotions' is a dict mapping emotion names to probabilities.
        """
        # Detect faces if not provided
        if face_rectangles is None:
            face_rectangles = self.detect_faces(image)
        
        if not face_rectangles:
            return []
        
        results = []
        
        # Process each face
        for box in face_rectangles:
            try:
                face_tensor = self._preprocess_face(image, box)
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                
                emotions = {
                    label: float(prob)
                    for label, prob in zip(EMOTION_LABELS, probabilities.cpu().numpy())
                }
                
                results.append({
                    "box": list(box),
                    "emotions": emotions,
                })
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return results
    
    def top_emotion(self, image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Get the top emotion for the first face in the image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Tuple of (emotion_name, probability) or (None, None) if no face detected
        """
        results = self.detect_emotions(image)
        
        if not results:
            return None, None
        
        emotions = results[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        
        return top_emotion, emotions[top_emotion]


# For backwards compatibility, create an alias
FineTunedFER = EmotionDetector


if __name__ == "__main__":
    # Quick test
    import sys
    
    detector = EmotionDetector()
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("\nTesting with webcam (press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = detector.detect_emotions(frame)
            
            for face in results:
                x, y, w, h = face["box"]
                emotions = face["emotions"]
                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion]
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{top_emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Emotion Detection (Fine-tuned)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")

