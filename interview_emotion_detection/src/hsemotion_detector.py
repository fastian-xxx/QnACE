"""
HSEmotion-based emotion detector using ONNX runtime for stability.

HSEmotion achieves 75%+ accuracy on FER2013, significantly better than 
training from scratch.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

# Try to import HSEmotion ONNX
try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
    print("Using HSEmotion ONNX backend")
except ImportError:
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer
        HSEMOTION_AVAILABLE = True
        print("Using HSEmotion PyTorch backend")
    except ImportError:
        HSEMOTION_AVAILABLE = False
        print("HSEmotion not available. Install with: pip install hsemotion-onnx")

# Try to import MTCNN for face detection
try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# Map HSEmotion labels to our standard labels
HSEMOTION_TO_STANDARD = {
    'Anger': 'angry',
    'Disgust': 'disgust', 
    'Fear': 'fear',
    'Happiness': 'happy',
    'Sadness': 'sad',
    'Surprise': 'surprise',
    'Neutral': 'neutral',
    # Some models use different labels
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'neutral': 'neutral',
}

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class HSEmotionDetector:
    """
    Emotion detector using HSEmotion pretrained models.
    
    HSEmotion provides state-of-the-art accuracy (75%+) on FER2013.
    """
    
    def __init__(self, model_name: str = 'enet_b0_8_best_afew', use_mtcnn: bool = True):
        """
        Initialize HSEmotion detector.
        
        Args:
            model_name: HSEmotion model to use. Options:
                - 'enet_b0_8_best_afew' (default, good accuracy)
                - 'enet_b0_8_best_vgaf'
                - 'enet_b2_8'
            use_mtcnn: Whether to use MTCNN for face detection
        """
        if not HSEMOTION_AVAILABLE:
            raise ImportError("HSEmotion not installed. Run: pip install hsemotion-onnx")
        
        # Initialize HSEmotion recognizer
        self.recognizer = HSEmotionRecognizer(model_name=model_name)
        print(f"Loaded HSEmotion model: {model_name}")
        
        # Setup face detector
        self.use_mtcnn = use_mtcnn and MTCNN_AVAILABLE
        if self.use_mtcnn:
            device = 'cpu'  # Use CPU for MTCNN with ONNX
            self.face_detector = MTCNN(keep_all=True, device=device)
            print("Using MTCNN for face detection")
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            print("Using Haar cascade for face detection")
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, probs = self.face_detector.detect(image_rgb)
        
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                w, h = x2 - x1, y2 - y1
                faces.append((x1, y1, w, h))
        
        return faces
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image."""
        if self.use_mtcnn:
            return self._detect_faces_mtcnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def detect_emotions(
        self,
        image: np.ndarray,
        face_rectangles: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict]:
        """
        Detect emotions in faces within an image.
        
        Args:
            image: BGR image as numpy array
            face_rectangles: Optional list of (x, y, w, h) face boxes
        
        Returns:
            List of dicts with 'box' and 'emotions' keys for each face
        """
        # Detect faces if not provided
        if face_rectangles is None:
            face_rectangles = self.detect_faces(image)
        
        if not face_rectangles:
            return []
        
        # Convert BGR to RGB for HSEmotion
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        results = []
        
        for box in face_rectangles:
            try:
                x, y, w, h = box
                
                # Add padding
                pad = int(0.1 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                
                face = image_rgb[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                # Get emotion predictions from HSEmotion
                emotion, scores = self.recognizer.predict_emotions(face, logits=True)
                
                # Convert scores to probabilities using softmax
                scores = np.array(scores)
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
                
                # Get emotion labels from HSEmotion (8 emotions including Contempt)
                emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
                
                # Map to standard labels
                emotions = {}
                for label, prob in zip(emotion_labels, probs):
                    std_label = HSEMOTION_TO_STANDARD.get(label, label.lower())
                    if std_label in EMOTION_LABELS:
                        emotions[std_label] = float(prob)
                
                # Ensure all emotions are present
                for label in EMOTION_LABELS:
                    if label not in emotions:
                        emotions[label] = 0.0
                
                results.append({
                    "box": list(box),
                    "emotions": emotions,
                })
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return results
    
    def top_emotion(self, image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Get the top emotion for the first face in the image."""
        results = self.detect_emotions(image)
        
        if not results:
            return None, None
        
        emotions = results[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        
        return top_emotion, emotions[top_emotion]


if __name__ == "__main__":
    # Quick test with webcam
    print("\n" + "="*60)
    print("HSEmotion Detector Test (ONNX)")
    print("="*60)
    
    detector = HSEmotionDetector()
    
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
                
                # Show all emotions
                y_offset = y + h + 20
                for emo, prob in sorted(emotions.items(), key=lambda x: -x[1]):
                    bar_width = int(prob * 100)
                    cv2.rectangle(frame, (x, y_offset), (x + bar_width, y_offset + 12), (0, 200, 0), -1)
                    cv2.putText(frame, f"{emo}: {prob:.2f}", (x + 105, y_offset + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 18
            
            cv2.imshow("HSEmotion Detection (75%+ accuracy)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")
