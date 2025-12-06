# src/interview_realtime_system.py
import cv2
import time
import os
import numpy as np
from interview_analyzer import InterviewEmotionAnalyzer
from interview_report_generator import InterviewReportGenerator

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to use fine-tuned model, fall back to FER if not available
USE_FINETUNED_MODEL = True

class InterviewRealtimeSystem:
    def __init__(self, use_finetuned: bool = USE_FINETUNED_MODEL):
        print("Initializing Complete Interview Emotion Detection System...")
        
        # Initialize core components
        if use_finetuned:
            try:
                from emotion_detector import EmotionDetector
                self.detector = EmotionDetector(use_mtcnn=True)
                print("Using FINE-TUNED emotion model (69.9% accuracy)")
            except Exception as e:
                print(f"Could not load fine-tuned model: {e}")
                print("Falling back to default FER model")
                from fer import FER
                self.detector = FER(mtcnn=True)
        else:
            from fer import FER
            self.detector = FER(mtcnn=True)
            print("Using default FER model (60% accuracy)")
        
        self.analyzer = InterviewEmotionAnalyzer(window_size=15)
        self.report_generator = InterviewReportGenerator()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection state management
        self.last_emotions_result = None
        self.last_analysis = None
        self.detection_frequency = 5  # Process every 5th frame
        
        # Interview session management
        self.session_active = False
        self.session_start_time = None
        self.session_data_file = None
        
        # UI display settings
        self.show_debug_info = False
        self.show_detailed_emotions = True
        
        print("System initialized successfully!")
    
    def calculate_fps(self):
        """Calculate and update current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_interview_overlay(self, frame, emotions_result, analysis=None):
        """Draw comprehensive interview analysis overlay"""
        
        # Use cached result if current detection failed (reduces flickering)
        if not emotions_result and self.last_emotions_result:
            emotions_result = self.last_emotions_result
            if self.show_debug_info:
                cv2.putText(frame, "(cached detection)", (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Handle no face detection
        if not emotions_result:
            self.draw_no_face_overlay(frame)
            return
        
        # Store successful result
        self.last_emotions_result = emotions_result
        
        # Get face data
        face_data = emotions_result[0]
        box = face_data['box']
        emotions = face_data['emotions']
        
        # Draw face detection box
        self.draw_face_box(frame, box, emotions)
        
        # Draw appropriate analysis overlay
        if analysis and self.session_active:
            self.draw_interview_analysis_overlay(frame, analysis, emotions, box)
        else:
            self.draw_basic_emotions_overlay(frame, emotions)
    
    def draw_no_face_overlay(self, frame):
        """Draw overlay when no face is detected"""
        cv2.putText(frame, "No face detected", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Position yourself clearly in camera view", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "Ensure good lighting and face the camera", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def draw_face_box(self, frame, box, emotions):
        """Draw face detection box with dominant emotion"""
        x, y, w, h = box
        
        # Get dominant emotion for coloring
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_confidence = emotions[dominant_emotion]
        
        # Color based on dominant emotion
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'neutral': (0, 255, 255),  # Yellow
            'surprise': (0, 165, 255), # Orange
            'sad': (255, 0, 0),        # Blue
            'fear': (128, 0, 128),     # Purple
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 128)   # Teal
        }
        
        box_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        
        # Draw dominant emotion label on face
        label = f"{dominant_emotion}: {dominant_confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    
    def draw_interview_analysis_overlay(self, frame, analysis, emotions, box):
        """Draw comprehensive interview analysis during active session"""
        
        # Main confidence display
        confidence = analysis['smoothed_confidence']
        level = analysis['confidence_level']
        color = analysis['color']
        
        # Large confidence header
        cv2.putText(frame, "INTERVIEW ANALYSIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Confidence: {level}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Score: {confidence:.3f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence progress bar
        bar_width = int(confidence * 300)
        cv2.rectangle(frame, (10, 100), (10 + bar_width, 115), color, -1)
        cv2.rectangle(frame, (10, 100), (310, 115), (255, 255, 255), 2)
        
        # Add percentage text on bar
        cv2.putText(frame, f"{confidence*100:.1f}%", (320, 112), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Real-time advice
        advice = analysis['advice']
        cv2.putText(frame, f"Advice: {advice[:50]}", (10, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(advice) > 50:
            cv2.putText(frame, advice[50:100], (10, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stability and trend indicators
        stability = analysis['stability']
        trend = stability['trend']
        consistency = stability['consistency']
        
        trend_colors = {
            'improving': (0, 255, 0),
            'stable': (0, 255, 255),
            'declining': (0, 100, 255)
        }
        trend_color = trend_colors.get(trend, (255, 255, 255))
        
        cv2.putText(frame, f"Trend: {trend.upper()}", (10, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, trend_color, 1)
        
        cv2.putText(frame, f"Consistency: {consistency:.2f}", (10, 195), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current emotions display (right side)
        if self.show_detailed_emotions:
            self.draw_emotion_breakdown(frame, emotions)
        
        # Session duration
        if self.session_active:
            duration = time.time() - self.session_start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", 
                       (frame.shape[1] - 150, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_basic_emotions_overlay(self, frame, emotions):
        """Draw basic emotion display when session is not active"""
        cv2.putText(frame, "EMOTION DETECTION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press SPACE to start interview session", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, confidence) in enumerate(sorted_emotions[:3]):
            # Color based on confidence level
            if confidence > 0.5:
                color = (0, 255, 0)      # Green - high confidence
            elif confidence > 0.3:
                color = (0, 255, 255)    # Yellow - medium
            else:
                color = (200, 200, 200)  # Gray - low
            
            text = f"{emotion}: {confidence:.3f} ({confidence*100:.1f}%)"
            y_pos = 90 + (i * 30)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_emotion_breakdown(self, frame, emotions):
        """Draw detailed emotion breakdown on right side"""
        start_x = frame.shape[1] - 280
        cv2.putText(frame, "CURRENT EMOTIONS", (start_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, confidence) in enumerate(sorted_emotions):
            y_pos = 55 + (i * 25)
            
            # Confidence bar
            bar_length = int(confidence * 150)
            cv2.rectangle(frame, (start_x, y_pos - 8), (start_x + bar_length, y_pos - 3), 
                         (100, 100, 100), -1)
            
            # Emotion text
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (start_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_system_status(self, frame):
        """Draw system status indicators"""
        
        # Session status indicator
        if self.session_active:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, "RECORDING", (frame.shape[1] - 120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (100, 100, 100), -1)
            cv2.putText(frame, "READY", (frame.shape[1] - 80, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Performance information
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (frame.shape[1] - 100, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if self.show_debug_info:
            cv2.putText(frame, f"Detection: 1/{self.detection_frequency} frames", 
                       (frame.shape[1] - 220, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(frame, f"Frames analyzed: {self.analyzer.total_frames_analyzed}", 
                       (frame.shape[1] - 220, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print("INTERVIEW EMOTION DETECTION SYSTEM - CONTROLS")
        print("="*60)
        print("SESSION CONTROLS:")
        print("  SPACE    - Start/Stop interview session")
        print("  's'      - Save current frame")
        print("  'r'      - Generate text report")
        print("  'v'      - Generate visual report with graphs")
        print()
        print("DISPLAY CONTROLS:")
        print("  'f'      - Toggle camera flip")
        print("  'd'      - Toggle debug information")
        print("  'e'      - Toggle detailed emotion display")
        print()
        print("SYSTEM CONTROLS:")
        print("  '+'      - Increase detection frequency")
        print("  '-'      - Decrease detection frequency")
        print("  'q'      - Quit application")
        print("="*60)
    
    def start_session(self):
        """Start interview session recording"""
        self.session_active = True
        self.session_start_time = time.time()
        
        # Reset analyzer for new session
        self.analyzer = InterviewEmotionAnalyzer(window_size=15)
        
        print("\nInterview session started!")
        print("Practice your interview responses while the system analyzes your expressions.")
    
    def stop_session(self):
        """Stop interview session recording"""
        self.session_active = False
        
        # Save session data
        if self.analyzer.total_frames_analyzed > 0:
            self.session_data_file = self.analyzer.save_session_data()
            print(f"\nInterview session stopped!")
            print(f"Session data saved. Use 'r' for text report or 'v' for visual report.")
        else:
            print("Session stopped - no data collected.")
    
    def generate_text_report(self):
        """Generate and display text report"""
        if self.analyzer.total_frames_analyzed == 0:
            print("No session data available for report generation.")
            return
        
        summary = self.analyzer.get_session_summary()
        
        print(f"\n" + "="*50)
        print("INTERVIEW SESSION ANALYSIS REPORT")
        print("="*50)
        
        # Session overview
        duration_min = int(summary['session_duration'] // 60)
        duration_sec = int(summary['session_duration'] % 60)
        print(f"Session Duration: {duration_min}m {duration_sec}s")
        print(f"Frames Analyzed: {summary['frames_analyzed']}")
        print(f"Average FPS: {summary['frames_analyzed']/summary['session_duration']:.1f}")
        
        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"Average Score: {summary['average_confidence']:.3f}")
        print(f"Score Range: {summary['min_confidence']:.3f} - {summary['max_confidence']:.3f}")
        
        level, message, advice, color = summary['final_assessment']
        print(f"Overall Level: {level}")
        print(f"Assessment: {message}")
        
        print(f"\nEMOTION BREAKDOWN:")
        emotion_dist = summary['emotion_distribution']
        total_frames = sum(emotion_dist.values())
        
        for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
            print(f"{emotion.capitalize():>10}: {count:>3} frames ({percentage:>5.1f}%)")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"- {advice}")
        
        # Additional insights based on data
        if summary['most_common_emotion'] == 'neutral':
            print("- Good professional composure maintained")
        
        if summary['average_confidence'] > 0.7:
            print("- Excellent interview presence detected")
        elif summary['average_confidence'] > 0.5:
            print("- Good confidence level with room for minor improvements")
        else:
            print("- Consider additional practice to improve confidence")
        
        print("="*50)
    
    def generate_visual_report(self):
        """Generate visual report with graphs"""
        if self.analyzer.total_frames_analyzed == 0:
            print("No session data available for visual report generation.")
            return
        
        print("Generating visual report...")
        summary = self.analyzer.get_session_summary()
        
        try:
            filename = self.report_generator.generate_visual_report(summary)
            print(f"Visual report generated successfully!")
            return filename
        except Exception as e:
            print(f"Error generating visual report: {e}")
            return None
    
    def run(self, flip_camera=True):
        """Main interview detection loop"""
        print("Starting Complete Interview Emotion Detection System...")
        self.print_controls()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Apply camera flip if enabled
                if flip_camera:
                    frame = cv2.flip(frame, 1)
                
                frame_count += 1
                self.calculate_fps()
                
                # Process emotions periodically for performance
                emotions_result = None
                analysis = None
                
                if frame_count % self.detection_frequency == 0:
                    try:
                        emotions_result = self.detector.detect_emotions(frame)
                        
                        # If session is active and we have emotions, analyze them
                        if self.session_active and emotions_result:
                            face_emotions = emotions_result[0]['emotions']
                            analysis = self.analyzer.add_emotion_data(face_emotions)
                            self.last_analysis = analysis
                        elif self.last_analysis:
                            # Use last analysis for display consistency
                            analysis = self.last_analysis
                            
                    except Exception as e:
                        if self.show_debug_info:
                            print(f"Detection error: {e}")
                
                # Draw all overlays
                self.draw_interview_overlay(frame, emotions_result, analysis)
                self.draw_system_status(frame)
                
                # Display frame
                cv2.imshow('Interview Emotion Detection System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space to start/stop session
                    if self.session_active:
                        self.stop_session()
                    else:
                        self.start_session()
                elif key == ord('s'):
                    filename = f"../outputs/interview_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved: {filename}")
                elif key == ord('r'):
                    self.generate_text_report()
                elif key == ord('v'):
                    self.generate_visual_report()
                elif key == ord('f'):
                    flip_camera = not flip_camera
                    print(f"Camera flip: {'ON' if flip_camera else 'OFF'}")
                elif key == ord('d'):
                    self.show_debug_info = not self.show_debug_info
                    print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
                elif key == ord('e'):
                    self.show_detailed_emotions = not self.show_detailed_emotions
                    print(f"Detailed emotions: {'ON' if self.show_detailed_emotions else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    self.detection_frequency = max(1, self.detection_frequency - 1)
                    print(f"Detection frequency increased: every {self.detection_frequency} frames")
                elif key == ord('-') or key == ord('_'):
                    self.detection_frequency = min(10, self.detection_frequency + 1)
                    print(f"Detection frequency decreased: every {self.detection_frequency} frames")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final report if session was active
            if self.session_active or self.analyzer.total_frames_analyzed > 0:
                print("\nGenerating final session report...")
                self.generate_text_report()
                
                # Ask if user wants visual report
                if self.analyzer.total_frames_analyzed > 10:  # Only if sufficient data
                    response = input("Generate visual report? (y/n): ").lower().strip()
                    if response.startswith('y'):
                        self.generate_visual_report()
            
            print("Interview Emotion Detection System closed.")

if __name__ == "__main__":
    # Create and run the system
    system = InterviewRealtimeSystem()
    system.run(flip_camera=True)