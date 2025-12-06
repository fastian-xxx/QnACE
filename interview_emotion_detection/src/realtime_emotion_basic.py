# src/realtime_emotion_basic.py
import cv2
from fer import FER
import time
import os

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicRealtimeEmotionDetector:
    def __init__(self):
        print("Initializing real-time emotion detector...")
        self.detector = FER(mtcnn=True)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_fps(self):
        """Calculate and display current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_emotion_overlay(self, frame, emotions_result):
        """Draw emotion information on the frame"""
        if not emotions_result:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        # Process first detected face
        face_data = emotions_result[0]
        box = face_data['box']
        emotions = face_data['emotions']
        
        # Draw face rectangle
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Draw emotion information
        for i, (emotion, confidence) in enumerate(sorted_emotions[:3]):
            # Color coding
            if confidence > 0.5:
                color = (0, 255, 0)      # Green - high confidence
            elif confidence > 0.3:
                color = (0, 255, 255)    # Yellow - medium
            else:
                color = (255, 255, 255)  # White - low
            
            # Display text
            text = f"{emotion}: {confidence:.3f}"
            y_pos = 30 + (i * 25)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show dominant emotion on face
        dominant_emotion = sorted_emotions[0][0]
        dominant_confidence = sorted_emotions[0][1]
        cv2.putText(frame, f"{dominant_emotion}: {dominant_confidence:.2f}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self):
        """Main detection loop"""
        print("Starting real-time emotion detection...")
        print("Controls: 'q' to quit, 's' to save frame")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            self.calculate_fps()
            
            # Process emotions every few frames for performance
            if frame_count % 3 == 0:  # Process every 3rd frame
                try:
                    emotions_result = self.detector.detect_emotions(frame)
                except Exception as e:
                    print(f"Detection error: {e}")
                    emotions_result = None
            
            # Draw overlay
            self.draw_emotion_overlay(frame, emotions_result if frame_count % 3 == 0 else None)
            
            # Show FPS
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                       (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"../outputs/realtime_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped")

if __name__ == "__main__":
    detector = BasicRealtimeEmotionDetector()
    detector.run()