# src/realtime_emotion_fixed.py
import cv2
from fer import FER
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FixedRealtimeEmotionDetector:
    def __init__(self):
        print("Initializing fixed real-time emotion detector...")
        self.detector = FER(mtcnn=True)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_emotions_result = None  # Store last result to avoid "no face" flicker
        
    def calculate_fps(self):
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_emotion_overlay(self, frame, emotions_result):
        """Draw emotion information with better handling"""
        
        # Use last result if current detection failed (reduces flickering)
        if not emotions_result and self.last_emotions_result:
            emotions_result = self.last_emotions_result
            # Add indicator that we're using cached result
            cv2.putText(frame, "(cached)", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        if not emotions_result:
            cv2.putText(frame, "No face detected - move closer", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        # Store successful result
        self.last_emotions_result = emotions_result
        
        # Process first detected face
        face_data = emotions_result[0]
        box = face_data['box']
        emotions = face_data['emotions']
        
        # Draw face rectangle
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Draw emotion information with better formatting
        for i, (emotion, confidence) in enumerate(sorted_emotions[:3]):
            # Color coding based on confidence
            if confidence > 0.5:
                color = (0, 255, 0)      # Green - high confidence
            elif confidence > 0.3:
                color = (0, 255, 255)    # Yellow - medium
            else:
                color = (200, 200, 200)  # Light gray - low
            
            # Display text with percentage
            text = f"{emotion}: {confidence:.3f} ({confidence*100:.1f}%)"
            y_pos = 30 + (i * 30)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add confidence bar
            bar_length = int(confidence * 200)  # Scale to 200 pixels max
            cv2.rectangle(frame, (250, y_pos - 15), (250 + bar_length, y_pos - 5), color, -1)
        
        # Show dominant emotion on face with larger text
        dominant_emotion = sorted_emotions[0][0]
        dominant_confidence = sorted_emotions[0][1]
        cv2.putText(frame, f"{dominant_emotion}: {dominant_confidence:.2f}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def run(self, flip_camera=True):
        """Main detection loop with camera flip option"""
        print("Starting fixed real-time emotion detection...")
        print("Controls:")
        print("  'q' - quit")
        print("  's' - save frame") 
        print("  'f' - toggle camera flip")
        print("  'r' - reset detection")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        detection_frequency = 5  # Process every 5th frame initially
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip camera horizontally to fix mirror effect
            if flip_camera:
                frame = cv2.flip(frame, 1)
            
            frame_count += 1
            self.calculate_fps()
            
            emotions_result = None
            
            # Process emotions periodically for performance
            if frame_count % detection_frequency == 0:
                try:
                    print(f"Processing frame {frame_count}...")  # Debug output
                    emotions_result = self.detector.detect_emotions(frame)
                    if emotions_result:
                        print(f"Found {len(emotions_result)} face(s)")  # Debug output
                    else:
                        print("No faces detected this frame")  # Debug output
                except Exception as e:
                    print(f"Detection error: {e}")
                    emotions_result = None
            
            # Draw overlay
            self.draw_emotion_overlay(frame, emotions_result)
            
            # Show performance info
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                       (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Processing every {detection_frequency} frames", 
                       (frame.shape[1] - 280, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow('Fixed Real-time Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"../outputs/realtime_fixed_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
            elif key == ord('f'):
                flip_camera = not flip_camera
                print(f"Camera flip: {'ON' if flip_camera else 'OFF'}")
            elif key == ord('r'):
                self.last_emotions_result = None
                print("Detection cache reset")
            elif key == ord('+'):
                detection_frequency = max(1, detection_frequency - 1)
                print(f"Detection frequency increased: every {detection_frequency} frames")
            elif key == ord('-'):
                detection_frequency = min(10, detection_frequency + 1)
                print(f"Detection frequency decreased: every {detection_frequency} frames")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Fixed real-time detection stopped")

if __name__ == "__main__":
    detector = FixedRealtimeEmotionDetector()
    detector.run(flip_camera=True)  # Start with camera flipped