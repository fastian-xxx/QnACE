# src/understand_fer_detailed.py
from fer import FER
import cv2
import numpy as np
import json
import time
from pprint import pprint

class FERAnalyzer:
    def __init__(self):
        print("🤖 Initializing FER detector...")
        self.detector = FER(mtcnn=True)
        print("✅ FER detector ready!")
        
    def analyze_single_frame(self):
        """Capture and analyze a single frame with detailed explanation"""
        
        print("\n📸 SINGLE FRAME ANALYSIS")
        print("=" * 40)
        
        # Capture frame
        cap = cv2.VideoCapture(0)
        print("📷 Position yourself in front of the camera...")
        print("Press SPACE when ready, or 'q' to skip")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera error")
                return
            
            # Show live preview
            cv2.putText(frame, "Press SPACE to capture, Q to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Position Yourself', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze the captured frame
        print("\n🔍 Analyzing your expression...")
        start_time = time.time()
        result = self.detector.detect_emotions(frame)
        analysis_time = time.time() - start_time
        
        print(f"⏱️ Analysis took: {analysis_time:.3f} seconds")
        
        # Detailed output explanation
        print("\n📊 RAW FER OUTPUT:")
        print("-" * 20)
        pprint(result)
        
        if not result:
            print("\n❌ No faces detected!")
            print("💡 Tips:")
            print("   - Make sure you're well-lit")
            print("   - Face the camera directly")
            print("   - Move closer to the camera")
            return
        
        # Explain each part of the output
        print(f"\n📝 OUTPUT EXPLANATION:")
        print(f"   📦 Type: {type(result)} (list of detected faces)")
        print(f"   👥 Number of faces: {len(result)}")
        
        # Analyze first face
        face_data = result[0]
        print(f"\n🎭 FACE 1 ANALYSIS:")
        
        # Bounding box
        box = face_data['box']
        print(f"   📍 Bounding Box: {box}")
        print(f"      x={box[0]}, y={box[1]}, width={box[2]}, height={box[3]}")
        print(f"      Face area: {box[2] * box[3]} pixels")
        
        # Emotions breakdown
        emotions = face_data['emotions']
        print(f"\n   🎭 EMOTIONS BREAKDOWN:")
        
        # Sort emotions by confidence
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, confidence) in enumerate(sorted_emotions):
            percentage = confidence * 100
            # Create visual bar
            bar_length = int(percentage / 2)  # Scale to reasonable length
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            # Color coding for terminal
            if i == 0:  # Dominant emotion
                status = "🥇 DOMINANT"
            elif confidence > 0.1:
                status = "📊 NOTABLE"
            else:
                status = "📉 LOW"
            
            print(f"      {emotion:8} | {confidence:.4f} ({percentage:5.1f}%) | {bar} | {status}")
        
        # Calculate emotion categories
        positive_emotions = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.7
        negative_emotions = (emotions.get('sad', 0) + emotions.get('fear', 0) + 
                           emotions.get('angry', 0) + emotions.get('disgust', 0))
        neutral_emotion = emotions.get('neutral', 0)
        
        print(f"\n   📈 EMOTION CATEGORIES:")
        print(f"      Positive: {positive_emotions:.3f}")
        print(f"      Negative: {negative_emotions:.3f}")
        print(f"      Neutral:  {neutral_emotion:.3f}")
        
        # Interview confidence assessment
        self.assess_interview_confidence(emotions)
        
        # Save analyzed image
        self.save_analyzed_image(frame, face_data)
        
        return result
    
    def assess_interview_confidence(self, emotions):
        """Assess interview confidence based on emotions"""
        
        print(f"\n💼 INTERVIEW CONFIDENCE ASSESSMENT:")
        
        # Interview-specific emotion weights
        confidence_weights = {
            'happy': 0.8,      # Positive indicator
            'neutral': 0.6,    # Professional composure
            'surprise': 0.3,   # Could be engagement
            'sad': -0.4,       # Low confidence indicator
            'fear': -0.7,      # High anxiety indicator
            'angry': -0.3,     # Could show frustration
            'disgust': -0.2    # Mild negative indicator
        }
        
        # Calculate weighted confidence score
        confidence_score = 0
        for emotion, weight in confidence_weights.items():
            emotion_prob = emotions.get(emotion, 0)
            contribution = emotion_prob * weight
            confidence_score += contribution
            
            if emotion_prob > 0.1:  # Only show significant contributors
                print(f"      {emotion:8}: {emotion_prob:.3f} × {weight:5.1f} = {contribution:6.3f}")
        
        # Normalize to 0-1 range
        confidence_score = max(0, min(1, (confidence_score + 1) / 2))
        
        print(f"   📊 Raw Score: {confidence_score:.3f}")
        
        # Provide assessment
        if confidence_score >= 0.8:
            level = "EXCELLENT"
            color = "🟢"
            advice = "You appear very confident and composed!"
        elif confidence_score >= 0.6:
            level = "GOOD"
            color = "🟡"
            advice = "Good confidence level with minor room for improvement"
        elif confidence_score >= 0.4:
            level = "MODERATE"
            color = "🟠"
            advice = "Some nervousness visible, practice relaxation techniques"
        else:
            level = "NEEDS WORK"
            color = "🔴"
            advice = "High anxiety detected, consider mock interview practice"
        
        print(f"   {color} Confidence Level: {level}")
        print(f"   💡 Advice: {advice}")
    
    def save_analyzed_image(self, frame, face_data):
        """Save image with emotion analysis overlay"""
        
        # Create annotated image
        annotated = frame.copy()
        
        # Draw bounding box
        box = face_data['box']
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get top 3 emotions
        emotions = face_data['emotions']
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Add emotion labels
        for i, (emotion, confidence) in enumerate(sorted_emotions[:3]):
            label = f"{emotion}: {confidence:.3f}"
            y_pos = y - 30 + (i * 20)
            
            # Choose color based on rank
            colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0)]  # Green, Yellow, Cyan
            color = colors[i] if i < len(colors) else (255, 255, 255)
            
            cv2.putText(annotated, label, (x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save image
        import os
        os.makedirs('../outputs', exist_ok=True)
        filename = f"../outputs/emotion_analysis_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"\n💾 Analyzed image saved to: {filename}")
    
    def compare_multiple_expressions(self):
        """Compare different facial expressions"""
        
        expressions_to_test = [
            ("😐 Neutral", "Keep a relaxed, professional face"),
            ("😊 Happy", "Give a genuine smile"),
            ("😟 Concerned", "Show slight worry or concern"),
            ("😮 Surprised", "Act surprised or engaged"),
            ("😤 Confident", "Look determined and confident")
        ]
        
        results = []
        
        print("\n🎭 EXPRESSION COMPARISON TEST")
        print("=" * 40)
        
        for i, (expression, instruction) in enumerate(expressions_to_test):
            print(f"\n{i+1}. Testing: {expression}")
            print(f"   📝 Instruction: {instruction}")
            print("   Press SPACE when ready...")
            
            # Capture expression
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                cv2.putText(frame, instruction, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to capture", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Expression Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
            
            cap.release()
            
            # Analyze
            result = self.detector.detect_emotions(frame)
            
            if result:
                emotions = result[0]['emotions']
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]
                
                results.append({
                    'intended': expression,
                    'detected': dominant,
                    'confidence': confidence,
                    'all_emotions': emotions
                })
                
                print(f"   ✅ Detected: {dominant} (confidence: {confidence:.3f})")
            else:
                print(f"   ❌ No face detected")
                results.append({
                    'intended': expression,
                    'detected': None,
                    'confidence': 0,
                    'all_emotions': {}
                })
        
        cv2.destroyAllWindows()
        
        # Print comparison summary
        print(f"\n📊 EXPRESSION TEST SUMMARY:")
        print("-" * 50)
        
        for result in results:
            intended = result['intended']
            detected = result['detected']
            confidence = result['confidence']
            
            if detected:
                match_status = "✅" if any(word in intended.lower() for word in detected.lower().split()) else "❌"
                print(f"{match_status} {intended:15} → {detected:8} ({confidence:.3f})")
            else:
                print(f"❌ {intended:15} → No detection")
        
        return results

def main():
    """Main function to run FER analysis"""
    
    analyzer = FERAnalyzer()
    
    print("🎯 FER DETAILED ANALYSIS TOOL")
    print("Choose an option:")
    print("1. Single frame analysis")
    print("2. Expression comparison test")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        analyzer.analyze_single_frame()
    
    if choice in ['2', '3']:
        time.sleep(2)  # Brief pause between tests
        analyzer.compare_multiple_expressions()
    
    print("\n🎉 FER analysis complete!")
    print("📁 Check the outputs folder for saved images")

if __name__ == "__main__":
    main()