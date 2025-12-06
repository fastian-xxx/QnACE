# src/understand_fer.py
from fer import FER
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def analyze_fer_output():
    """Understand what FER returns"""
    
    # Initialize FER detector
    detector = FER(mtcnn=True)  # mtcnn=True gives better face detection
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("📸 Taking a photo in 3 seconds...")
    print("Look at the camera and make different expressions!")
    
    # Wait 3 seconds
    for i in range(3, 0, -1):
        print(f"⏰ {i}...")
        ret, frame = cap.read()
        cv2.waitKey(1000)
    
    # Capture final frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not capture image")
        return
    
    # Analyze the image
    print("\n🔍 Analyzing your expression...")
    result = detector.detect_emotions(frame)
    
    print("\n📊 RAW OUTPUT:")
    pprint(result)
    
    # Explain the output structure
    if result:
        print("\n📝 EXPLANATION:")
        print("The output is a list of dictionaries, one for each detected face.")
        print("\nFor each face:")
        
        face = result[0]  # First (and usually only) face
        print(f"📍 'box': {face['box']} - [x, y, width, height] of face rectangle")
        print("🎭 'emotions': Dictionary with 7 emotion probabilities:")
        
        for emotion, prob in face['emotions'].items():
            percentage = prob * 100
            bar = "█" * int(percentage / 5)  # Visual bar
            print(f"   {emotion:8}: {prob:.3f} ({percentage:5.1f}%) {bar}")
        
        # Find dominant emotion
        dominant = max(face['emotions'], key=face['emotions'].get)
        confidence = face['emotions'][dominant]
        print(f"\n🎯 DOMINANT EMOTION: {dominant.upper()} (confidence: {confidence:.3f})")
        
        # Save the analyzed image
        cv2.rectangle(frame, (face['box'][0], face['box'][1]), 
                     (face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]), 
                     (0, 255, 0), 2)
        cv2.putText(frame, f"{dominant}: {confidence:.2f}", 
                   (face['box'][0], face['box'][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imwrite('../outputs/test_emotion_detection.jpg', frame)
        print(f"\n💾 Saved analyzed image to ../outputs/test_emotion_detection.jpg")
    else:
        print("❌ No faces detected in the image")

if __name__ == "__main__":
    analyze_fer_output()