# src/realtime_emotion.py
import cv2
from fer import FER
import time
import numpy as np

class RealtimeEmotionDetector:
    def __init__(self):
        print("🤖 Initializing emotion detector...")
        self.detector = FER(mtcnn=True)
        self.camera_index = self.find_camera()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection settings
        self.detection_frequency = 5  # Process every 5th frame for performance
        self.frame_count = 0
        self.last_emotions = None
        
    def find_camera(self):
        """Find working camera"""
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"✅ Using camera {i}")
                    return i
        print("❌ No camera found!")
        return None
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def process_emotions(self, frame):
        """Process emotions with performance optimization"""
        self.frame_count += 1
        
        # Only process every nth frame
        if self.frame_count % self.detection_frequency == 0:
            try:
                emotions_result = self.detector.detect_emotions(frame)
                if emotions_result:
                    self.last_emotions = emotions_result[0]  # Take first face
                    return self.last_emotions
            except Exception as e:
                print(f"⚠️  Emotion detection error: {e}")
        
        return self.last_emotions
    
    def draw_emotion_info(self, frame, emotion_data):
        """Draw emotion information on frame"""
        if not emotion_data:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        # Draw face rectangle
        box = emotion_data['box']
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Get emotions
        emotions = emotion_data['emotions']
        
        # Find top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_emotions[:3]
        
        # Draw emotion bars
        for i, (emotion, confidence) in enumerate(top_3):
            # Color based on confidence
            if confidence > 0.5:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.3:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 165, 255)  # Orange for low
            
            # Draw emotion name and confidence
            text = f"{emotion}: {confidence:.2f}"
            y_pos = 30 + (i * 25)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence bar
            bar_width = int(confidence * 200)  # Max 200 pixels
            cv2.rectangle(frame, (200, y_pos - 15), (200 + bar_width, y_pos - 5), color, -1)
        
        # Draw dominant emotion on face
        dominant_emotion = top_3[0][0]
        dominant_confidence = top_3[0][1]
        cv2.putText(frame, f"{dominant_emotion}: {dominant_confidence:.2f}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def draw_performance_info(self, frame):
        """Draw performance information"""
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Processing frequency
        cv2.putText(frame, f"Process every {self.detection_frequency} frames", 
                   (frame.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main detection loop"""
        if self.camera_index is None:
            return
        
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎥 Real-time emotion detection started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  '+' - Increase detection frequency")
        print("  '-' - Decrease detection frequency")
        print("  's' - Save current frame with analysis")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            self.calculate_fps()
            
            # Process emotions
            emotion_data = self.process_emotions(frame)
            
            # Draw information
            self.draw_emotion_info(frame, emotion_data)
            self.draw_performance_info(frame)
            
            # Show frame
            cv2.imshow('Real-time Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.detection_frequency = max(1, self.detection_frequency - 1)
                print(f"Detection frequency: every {self.detection_frequency} frames")
            elif key == ord('-'):
                self.detection_frequency = min(10, self.detection_frequency + 1)
                print(f"Detection frequency: every {self.detection_frequency} frames")
            elif key == ord('s'):
                filename = f"../outputs/emotion_analysis_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved analysis to {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Real-time detection stopped")

if __name__ == "__main__":
    detector = RealtimeEmotionDetector()
    detector.run()