"""
Q&ACE Integrated System - Quick Test

Run this to verify the multimodal system is working correctly.
"""

import sys
from pathlib import Path

# Add paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "interview_emotion_detection" / "src"))
sys.path.insert(0, str(ROOT_DIR / "integrated_system"))

import numpy as np


def test_facial_detector():
    """Test facial emotion detector."""
    print("\n" + "="*50)
    print("🎭 Testing Facial Emotion Detector")
    print("="*50)
    
    try:
        from emotion_detector import EmotionDetector
        
        detector = EmotionDetector()
        print("✅ Facial detector initialized!")
        
        # Create dummy image
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_emotions(dummy_frame)
        
        print(f"✅ Detection works! (No face in dummy image: {len(result)} faces)")
        return True
        
    except Exception as e:
        print(f"❌ Facial detector failed: {e}")
        return False


def test_voice_detector():
    """Test voice emotion detector."""
    print("\n" + "="*50)
    print("🎤 Testing Voice Emotion Detector")
    print("="*50)
    
    try:
        from voice_emotion_detector import VoiceEmotionDetector
        
        detector = VoiceEmotionDetector()
        print("✅ Voice detector initialized!")
        
        # Create dummy audio (2 seconds of silence)
        dummy_audio = np.zeros(32000, dtype=np.float32)
        result = detector.detect_emotions(dummy_audio)
        
        print(f"✅ Detection works!")
        print(f"   Dominant: {result['dominant_emotion']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        return True
        
    except FileNotFoundError as e:
        print(f"⚠️ Voice model not found: {e}")
        print("   Make sure QnAce_Voice-Model.pth is in the correct location")
        return False
    except Exception as e:
        print(f"❌ Voice detector failed: {e}")
        return False


def test_multimodal_detector():
    """Test multimodal detector."""
    print("\n" + "="*50)
    print("🎯 Testing Multimodal Detector")
    print("="*50)
    
    try:
        from multimodal_detector import MultimodalEmotionDetector
        
        detector = MultimodalEmotionDetector()
        print("✅ Multimodal detector initialized!")
        
        # Test with dummy data
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_audio = np.zeros(32000, dtype=np.float32)
        
        result = detector.detect(frame=dummy_frame, audio=dummy_audio)
        
        print(f"✅ Multimodal detection works!")
        print(f"   Fused Emotion: {result.dominant_emotion}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Face detected: {result.face_detected}")
        print(f"   Voice detected: {result.voice_detected}")
        return True
        
    except Exception as e:
        print(f"❌ Multimodal detector failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generator():
    """Test report generator."""
    print("\n" + "="*50)
    print("📊 Testing Report Generator")
    print("="*50)
    
    try:
        from report_generator import generate_reports
        
        # Sample data
        sample_data = {
            'session_id': 'TEST_001',
            'duration': 30.0,
            'avg_confidence': 65.0,
            'min_confidence': 45.0,
            'max_confidence': 82.0,
            'beginning_confidence': 55.0,
            'middle_confidence': 65.0,
            'end_confidence': 75.0,
            'emotion_distribution': {
                'neutral': 50.0,
                'happy': 30.0,
                'surprise': 10.0,
                'sad': 5.0,
                'fear': 5.0
            },
            'facial_frames': 80,
            'voice_frames': 15,
            'multimodal_frames': 10,
            'total_frames': 90,
            'frames': []
        }
        
        output_dir = str(ROOT_DIR / "outputs")
        reports = generate_reports(sample_data, output_dir)
        
        print(f"✅ Reports generated!")
        print(f"   PNG: {reports['png']}")
        print(f"   TXT: {reports['txt']}")
        return True
        
    except Exception as e:
        print(f"❌ Report generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 Q&ACE INTEGRATED SYSTEM - TEST SUITE")
    print("="*60)
    
    results = {
        'Facial Detector': test_facial_detector(),
        'Voice Detector': test_voice_detector(),
        'Multimodal Detector': test_multimodal_detector(),
        'Report Generator': test_report_generator()
    }
    
    print("\n" + "="*60)
    print("📋 TEST RESULTS")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test:25s} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("-"*60)
    print(f"  Total: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()
