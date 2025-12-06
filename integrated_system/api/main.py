"""
Q&ACE Unified API - Combines Facial, Voice, and Text (BERT) Analysis

This is the single API endpoint that the Frontend connects to.
It orchestrates all 3 ML models for comprehensive interview analysis.

Endpoints:
- POST /analyze/facial     - Analyze facial emotions from image
- POST /analyze/voice      - Analyze voice emotions from audio
- POST /analyze/text       - Analyze answer quality with BERT
- POST /analyze/multimodal - Analyze all modalities together
- GET  /health             - Health check
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64
import io
import tempfile

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "interview_emotion_detection" / "src"))
sys.path.insert(0, str(ROOT_DIR / "integrated_system"))

import torch

# ============================================
# FastAPI App Setup
# ============================================

app = FastAPI(
    title="Q&ACE Unified API",
    description="Multimodal Interview Analysis API - Facial, Voice, and Text",
    version="1.0.0",
)

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global Model Instances
# ============================================

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Model instances (lazy loaded)
facial_detector = None
voice_detector = None
bert_model = None
bert_tokenizer = None


def get_facial_detector():
    """Lazy load facial detector."""
    global facial_detector
    if facial_detector is None:
        try:
            from facial_emotion_detector import FacialEmotionDetector
            facial_detector = FacialEmotionDetector(device=str(DEVICE))
            print("✅ Facial detector loaded")
        except Exception as e:
            print(f"❌ Facial detector failed: {e}")
    return facial_detector


def get_voice_detector():
    """Lazy load voice detector."""
    global voice_detector
    if voice_detector is None:
        try:
            from voice_emotion_detector import VoiceEmotionDetector
            model_path = ROOT_DIR / "QnAce_Voice-Model" / "QnAce_Voice-Model.pth"
            voice_detector = VoiceEmotionDetector(
                model_path=str(model_path),
                device=str(DEVICE)
            )
            print("✅ Voice detector loaded")
        except Exception as e:
            print(f"❌ Voice detector failed: {e}")
    return voice_detector


def get_bert_model():
    """Lazy load BERT model."""
    global bert_model, bert_tokenizer
    if bert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_dir = ROOT_DIR / "BERT_Model"
            
            bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            bert_model.to(DEVICE)
            bert_model.eval()
            
            # Try local tokenizer, fallback to base
            try:
                bert_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            except:
                bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"❌ BERT model failed: {e}")
    return bert_model, bert_tokenizer


# ============================================
# Request/Response Models
# ============================================

class TextAnalysisRequest(BaseModel):
    text: str
    question: Optional[str] = None


class FacialAnalysisResponse(BaseModel):
    success: bool
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    face_detected: bool
    error: Optional[str] = None


class VoiceAnalysisResponse(BaseModel):
    success: bool
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    error: Optional[str] = None


class TextAnalysisResponse(BaseModel):
    success: bool
    quality_score: float  # 0-100
    quality_label: str    # "Poor", "Average", "Excellent"
    probabilities: Dict[str, float]
    feedback: str
    error: Optional[str] = None


class MultimodalAnalysisResponse(BaseModel):
    success: bool
    
    # Overall scores
    overall_confidence: float
    overall_emotion: str
    
    # Individual results
    facial: Optional[FacialAnalysisResponse] = None
    voice: Optional[VoiceAnalysisResponse] = None
    text: Optional[TextAnalysisResponse] = None
    
    # Fused emotions
    fused_emotions: Dict[str, float]
    
    # Interview metrics
    confidence_score: float  # 0-100
    clarity_score: float     # 0-100
    engagement_score: float  # 0-100
    
    # Recommendations
    recommendations: List[str]
    
    timestamp: str
    error: Optional[str] = None


# ============================================
# Utility Functions
# ============================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    import cv2
    
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def calculate_facial_score(facial_result: dict) -> float:
    """
    Calculate facial analysis score (0-100) optimized for interview performance.
    
    Professional interview expressions:
    - Neutral (professional, attentive) = Excellent (90-100)
    - Happy (confident, positive) = Excellent (90-100)  
    - Surprise (engaged, interested) = Good (75-85)
    - Sad/Fear (nervous) = Needs work (40-60)
    - Angry/Disgust (negative) = Poor (30-50)
    
    IMPORTANT: Includes "resting face correction" - many people with neutral/professional
    expressions are misclassified as "angry" by FER2013-trained models due to dataset bias.
    If angry is high but neutral+happy+surprise are also significant, it's likely a resting face.
    """
    if not facial_result or not facial_result.get('face_detected'):
        return 50.0  # Default if no face
    
    emotions = facial_result.get('emotions', {})
    confidence = facial_result.get('confidence', 0.5)
    
    # Get probabilities
    angry_prob = emotions.get('angry', 0)
    neutral_prob = emotions.get('neutral', 0)
    happy_prob = emotions.get('happy', 0)
    surprise_prob = emotions.get('surprise', 0)
    sad_prob = emotions.get('sad', 0)
    fear_prob = emotions.get('fear', 0)
    disgust_prob = emotions.get('disgust', 0)
    
    # ========== RESTING FACE CORRECTION ==========
    # FER2013 models often misclassify neutral/professional faces as "angry"
    
    positive_sum = neutral_prob + happy_prob + surprise_prob
    corrected_emotions = emotions.copy()
    
    if angry_prob > 0.35 and angry_prob < 0.70 and positive_sum > 0.15:
        # This is likely a resting face being misclassified
        correction_factor = min(0.6, positive_sum)
        angry_transfer = angry_prob * correction_factor
        
        corrected_emotions['neutral'] = neutral_prob + angry_transfer * 0.8
        corrected_emotions['happy'] = happy_prob + angry_transfer * 0.2
        corrected_emotions['angry'] = angry_prob * (1 - correction_factor)
        
        print(f"🔄 Resting face correction: angry {angry_prob:.1%} -> {corrected_emotions['angry']:.1%}")
    
    emotions = corrected_emotions
    
    # ========== REALISTIC EMOTION SCORING v2.0 ==========
    # Interview-appropriate scores - NOT inflated
    # Scale: 25-95 (never give 100% or below 20%)
    emotion_scores = {
        'happy': 85,      # Positive, confident - good
        'neutral': 70,    # Professional but not engaging - average
        'surprise': 55,   # Can seem unprepared - below average
        'sad': 35,        # Nervous, uncomfortable - poor
        'fear': 30,       # Very nervous - poor
        'angry': 40,      # Defensive/hostile - poor
        'disgust': 30,    # Very negative - poor
    }
    
    # Calculate weighted score
    weighted_score = 0
    total_weight = 0
    
    for emotion, prob in emotions.items():
        if emotion in emotion_scores and prob > 0.01:
            weighted_score += emotion_scores[emotion] * prob
            total_weight += prob
    
    base_score = weighted_score / total_weight if total_weight > 0 else 50
    
    # ========== PENALTIES AND BONUSES ==========
    
    # PENALTY: Low confidence = uncertain detection
    if confidence < 0.4:
        base_score -= 10
    elif confidence < 0.6:
        base_score -= 5
    
    # BONUS: High happy + neutral mix = ideal interview demeanor
    ideal_mix = happy_prob * 0.5 + neutral_prob * 0.5
    if ideal_mix > 0.5 and confidence > 0.5:
        base_score += 8
    
    # Cap the score between 20 and 95
    final_score = max(20, min(95, base_score))
    
    return round(final_score, 1)


def calculate_confidence_score(facial_result: dict, voice_result: dict, text_result: dict) -> float:
    """
    Calculate overall interview confidence score (0-100).
    
    REALISTIC SCORING v2.0:
    - Eyes closed + still face = LOW score (25-40%)
    - Nervous/fearful = LOW score (30-45%)
    - Neutral/professional = MEDIUM score (60-75%)
    - Happy/confident = HIGH score (75-90%)
    """
    scores = []
    weights = []
    
    # =====================================
    # FACIAL ANALYSIS (50% weight)
    # =====================================
    if facial_result and facial_result.get('face_detected'):
        facial_score = calculate_facial_score(facial_result)
        scores.append(facial_score)
        weights.append(0.50)
    else:
        # No face detected = severe penalty
        scores.append(25)
        weights.append(0.50)
    
    # =====================================
    # VOICE ANALYSIS (40% weight)
    # =====================================
    if voice_result and voice_result.get('emotions'):
        voice_emotions = voice_result.get('emotions', {})
        
        # Voice emotion scoring - similar logic to facial
        emotion_scores = {
            'neutral': 75,   # Professional tone
            'happy': 88,     # Confident, enthusiastic
            'surprise': 60,  # Can indicate uncertainty
            'sad': 40,       # Low energy, nervous
            'fear': 35,      # Clearly nervous
            'angry': 45,     # Too aggressive
            'disgust': 35,   # Negative
        }
        
        weighted_score = 0
        total_weight = 0
        for emotion, prob in voice_emotions.items():
            if emotion in emotion_scores and prob > 0.01:
                weighted_score += emotion_scores[emotion] * prob
                total_weight += prob
        
        voice_score = weighted_score / total_weight if total_weight > 0 else 60
        
        # Confidence adjustment
        voice_confidence = voice_result.get('confidence', 0.5)
        if voice_confidence < 0.4:
            voice_score -= 10
        elif voice_confidence > 0.7:
            voice_score += 5
        
        voice_score = min(95, max(25, voice_score))
        scores.append(voice_score)
        weights.append(0.40)
    else:
        # No voice analysis available
        scores.append(60)  # Neutral default
        weights.append(0.40)
    
    # =====================================
    # TEXT ANALYSIS (10% weight) - BERT is unreliable
    # =====================================
    if text_result:
        # BERT model is overfitting - use conservative scoring
        bert_score = text_result.get('quality_score', 50)
        # Trust BERT only slightly, default to reasonable score
        text_score = max(60, min(85, (60 + bert_score) / 2))
        scores.append(text_score)
        weights.append(0.10)
    
    if not scores:
        return 50.0
    
    # Calculate weighted average
    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    return round(min(100, max(0, weighted_score)), 1)


def generate_recommendations(facial_result: dict, voice_result: dict, text_result: dict) -> List[str]:
    """Generate personalized recommendations based on analysis."""
    recommendations = []
    
    # Facial recommendations
    if facial_result and facial_result.get('face_detected'):
        dominant = facial_result.get('dominant_emotion', '')
        if dominant in ['fear', 'sad']:
            recommendations.append("💡 Try to relax your facial expressions. Practice in front of a mirror to appear more confident.")
        elif dominant == 'angry':
            recommendations.append("💡 Soften your expressions. Take a deep breath before answering to appear more approachable.")
        elif dominant == 'neutral':
            recommendations.append("💡 Great neutral expression! Try adding a slight smile to appear more engaging.")
        elif dominant == 'happy':
            recommendations.append("✅ Excellent! Your positive expression conveys confidence and enthusiasm.")
    
    # Voice recommendations
    if voice_result and voice_result.get('emotions'):
        dominant = voice_result.get('dominant_emotion', '')
        if dominant in ['fear', 'sad']:
            recommendations.append("💡 Your voice indicates nervousness. Try speaking slightly slower and with more conviction.")
        elif dominant == 'anger':
            recommendations.append("💡 Your tone sounds intense. Try moderating your voice to sound more calm and professional.")
        elif dominant in ['happy', 'neutral']:
            recommendations.append("✅ Good vocal tone! You sound confident and professional.")
    
    # Text recommendations
    if text_result:
        quality = text_result.get('quality_label', '')
        if quality == 'Poor':
            recommendations.append("💡 Your answer could be more detailed. Use the STAR method (Situation, Task, Action, Result) for behavioral questions.")
        elif quality == 'Average':
            recommendations.append("💡 Good answer! Try adding more specific examples and quantifiable results to make it excellent.")
        elif quality == 'Excellent':
            recommendations.append("✅ Excellent answer! Well-structured with good detail and relevance.")
    
    if not recommendations:
        recommendations.append("💡 Keep practicing! Regular mock interviews will help you improve.")
    
    return recommendations


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Q&ACE Unified API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze/facial",
            "/analyze/voice", 
            "/analyze/text",
            "/analyze/multimodal",
            "/health"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "facial": facial_detector is not None,
            "voice": voice_detector is not None,
            "bert": bert_model is not None
        }
    }


@app.post("/analyze/facial", response_model=FacialAnalysisResponse)
async def analyze_facial(image: str = Form(...)):
    """
    Analyze facial emotions from base64 image.
    
    Args:
        image: Base64 encoded image string
    """
    try:
        detector = get_facial_detector()
        if detector is None:
            return FacialAnalysisResponse(
                success=False,
                emotions={},
                dominant_emotion="",
                confidence=0.0,
                face_detected=False,
                error="Facial detector not available"
            )
        
        # Decode image
        img = decode_base64_image(image)
        if img is None:
            return FacialAnalysisResponse(
                success=False,
                emotions={},
                dominant_emotion="",
                confidence=0.0,
                face_detected=False,
                error="Failed to decode image"
            )
        
        # Detect emotions (with fallback to center region if no face detected)
        result = detector.detect_emotions_from_frame(img, assume_face_if_not_detected=True)
        
        return FacialAnalysisResponse(
            success=True,
            emotions=result['emotions'],
            dominant_emotion=result['dominant_emotion'],
            confidence=result['confidence'],
            face_detected=result['face_detected']
        )
        
    except Exception as e:
        return FacialAnalysisResponse(
            success=False,
            emotions={},
            dominant_emotion="",
            confidence=0.0,
            face_detected=False,
            error=str(e)
        )


@app.post("/analyze/voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(audio: UploadFile = File(...)):
    """
    Analyze voice emotions from audio file.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
    """
    try:
        detector = get_voice_detector()
        if detector is None:
            return VoiceAnalysisResponse(
                success=False,
                emotions={},
                dominant_emotion="",
                confidence=0.0,
                error="Voice detector not available"
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load and analyze
            import librosa
            audio_data, sr = librosa.load(tmp_path, sr=16000)
            result = detector.detect_emotions(audio_data, sample_rate=sr)
            
            return VoiceAnalysisResponse(
                success=True,
                emotions=result['emotions'],
                dominant_emotion=result['dominant_emotion'],
                confidence=result['confidence']
            )
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        return VoiceAnalysisResponse(
            success=False,
            emotions={},
            dominant_emotion="",
            confidence=0.0,
            error=str(e)
        )


@app.post("/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze answer quality using BERT model.
    
    Args:
        request: Text analysis request with answer text
    """
    try:
        model, tokenizer = get_bert_model()
        if model is None or tokenizer is None:
            return TextAnalysisResponse(
                success=False,
                quality_score=0.0,
                quality_label="",
                probabilities={},
                feedback="",
                error="BERT model not available"
            )
        
        text = request.text.strip()
        if not text:
            return TextAnalysisResponse(
                success=False,
                quality_score=0.0,
                quality_label="",
                probabilities={},
                feedback="",
                error="Empty text provided"
            )
        
        # Tokenize
        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.detach().cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Map labels
        labels = {0: "Poor", 1: "Average", 2: "Excellent"}
        predicted_idx = int(np.argmax(probs))
        quality_label = labels[predicted_idx]
        
        # Calculate quality score (0-100)
        # Weight: Poor=0-33, Average=34-66, Excellent=67-100
        quality_score = (probs[0] * 16.5 + probs[1] * 50 + probs[2] * 83.5)
        
        # Generate feedback
        if quality_label == "Poor":
            feedback = "Your answer lacks depth and specificity. Try to include concrete examples and structure your response using the STAR method."
        elif quality_label == "Average":
            feedback = "Good foundation! To improve, add more specific details, quantifiable achievements, and connect your answer directly to the role requirements."
        else:
            feedback = "Excellent response! Well-structured with relevant details and clear communication. Keep up this level of preparation."
        
        return TextAnalysisResponse(
            success=True,
            quality_score=float(quality_score),
            quality_label=quality_label,
            probabilities={labels[i]: float(probs[i]) for i in range(3)},
            feedback=feedback
        )
        
    except Exception as e:
        return TextAnalysisResponse(
            success=False,
            quality_score=0.0,
            quality_label="",
            probabilities={},
            feedback="",
            error=str(e)
        )


@app.post("/analyze/multimodal", response_model=MultimodalAnalysisResponse)
async def analyze_multimodal(
    image: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    question: Optional[str] = Form(None)
):
    """
    Analyze all modalities together for comprehensive interview feedback.
    
    Args:
        image: Base64 encoded image (optional)
        audio: Audio file (optional)
        text: Answer text (optional)
        question: Interview question (optional)
    """
    try:
        facial_result = None
        voice_result = None
        text_result = None
        
        # Analyze facial
        if image:
            response = await analyze_facial(image=image)
            if response.success:
                facial_result = response.model_dump()
        
        # Analyze voice
        if audio:
            response = await analyze_voice(audio=audio)
            if response.success:
                voice_result = response.model_dump()
        
        # Analyze text
        if text:
            response = await analyze_text(TextAnalysisRequest(text=text, question=question))
            if response.success:
                text_result = response.model_dump()
        
        # Fuse emotions
        fused_emotions = {}
        emotion_sources = 0
        
        if facial_result and facial_result.get('face_detected'):
            for emotion, prob in facial_result.get('emotions', {}).items():
                fused_emotions[emotion] = fused_emotions.get(emotion, 0) + prob * 0.5
            emotion_sources += 1
        
        if voice_result:
            for emotion, prob in voice_result.get('emotions', {}).items():
                fused_emotions[emotion] = fused_emotions.get(emotion, 0) + prob * 0.5
            emotion_sources += 1
        
        # Normalize fused emotions
        if emotion_sources > 0:
            total = sum(fused_emotions.values())
            if total > 0:
                fused_emotions = {k: v/total for k, v in fused_emotions.items()}
        
        # Calculate scores
        confidence_score = calculate_confidence_score(facial_result or {}, voice_result or {}, text_result or {})
        
        # Clarity based on text quality
        clarity_score = text_result.get('quality_score', 50.0) if text_result else 50.0
        
        # Engagement based on emotion variety and positivity
        engagement_score = 50.0
        if fused_emotions:
            positive_emotions = sum(fused_emotions.get(e, 0) for e in ['happy', 'surprise'])
            engagement_score = min(100, 50 + positive_emotions * 50)
        
        # Get dominant emotion
        overall_emotion = "neutral"
        if fused_emotions:
            overall_emotion = max(fused_emotions, key=fused_emotions.get)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            facial_result or {},
            voice_result or {},
            text_result or {}
        )
        
        return MultimodalAnalysisResponse(
            success=True,
            overall_confidence=confidence_score,
            overall_emotion=overall_emotion,
            facial=FacialAnalysisResponse(**facial_result) if facial_result else None,
            voice=VoiceAnalysisResponse(**voice_result) if voice_result else None,
            text=TextAnalysisResponse(**text_result) if text_result else None,
            fused_emotions=fused_emotions,
            confidence_score=confidence_score,
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return MultimodalAnalysisResponse(
            success=False,
            overall_confidence=0.0,
            overall_emotion="",
            fused_emotions={},
            confidence_score=0.0,
            clarity_score=0.0,
            engagement_score=0.0,
            recommendations=[],
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Q&ACE Unified API Server                        ║
    ║                                                           ║
    ║   Facial + Voice + Text Analysis                          ║
    ║   http://localhost:8001                                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Pre-load models
    print("Loading models...")
    get_facial_detector()
    get_voice_detector()
    get_bert_model()
    print("\n✅ All models loaded! Starting server...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )
