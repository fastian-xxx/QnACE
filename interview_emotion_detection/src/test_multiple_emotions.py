# src/test_multiple_emotions.py
from fer import FER
import cv2
import numpy as np
import time

def capture_multiple_emotions():
    """Capture and analyze different emotions"""
    
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)
    
    # List of emotions to test
    emotions_to_test = [
        "😐 Neutral (relaxed face)",
        "😊 Happy (smile!)",
        "😟 Sad (frown)",
        "😠 Angry (scowl)",
        "😨 Fear (worried look)",
        "😮 Surprise (wide eyes, open mouth)",
        "🤢 Disgust (wrinkled nose)"
    ]
    
    results = []
    
    for i, emotion_prompt in enumerate(emotions_to_test):
        print(f"\n🎭 Test {i+1}/7: Make a {emotion_prompt}")
        print("Photo in 5 seconds...")
        
        # Countdown
        for countdown in range(5, 0, -1):
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"{emotion_prompt}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Photo in: {countdown}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Emotion Test', frame)
            cv2.waitKey(1000)
        
        # Capture and analyze
        ret, frame = cap.read()
        if ret:
            result = detector.detect_emotions(frame)
            if result:
                emotions_dict = result[0]['emotions']
                predicted = max(emotions_dict, key=emotions_dict.get)
                confidence = emotions_dict[predicted]
                
                results.append({
                    'intended': emotion_prompt.split()[0],
                    'predicted': predicted,
                    'confidence': confidence,
                    'all_emotions': emotions_dict
                })
                
                print(f"✅ Detected: {predicted} (confidence: {confidence:.3f})")
                
                # Save image
                filename = f"../outputs/emotion_test_{i+1}_{predicted}.jpg"
                cv2.imwrite(filename, frame)
            else:
                print("❌ No face detected")
        
        cv2.waitKey(2000)  # Pause between tests
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n📊 TEST SUMMARY:")
    print("-" * 50)
    for i, result in enumerate(results):
        intended = result['intended']
        predicted = result['predicted']
        confidence = result['confidence']
        match = "✅" if predicted.lower() in intended.lower() else "❌"
        print(f"{match} Test {i+1}: Intended {intended} → Predicted {predicted} ({confidence:.3f})")
    
    return results

if __name__ == "__main__":
    results = capture_multiple_emotions()