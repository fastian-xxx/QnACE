# src/interview_analyzer.py
"""
Enhanced Interview Emotion Analyzer with full data storage for accurate reporting.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class FrameData:
    """Data for a single analyzed frame."""
    timestamp: float
    emotions: Dict[str, float]
    confidence_score: float
    dominant_emotion: str
    face_detected: bool
    face_position: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class InterviewEmotionAnalyzer:
    """
    Enhanced interview emotion analyzer that stores ALL frame data
    for accurate reporting and analysis.
    """
    
    def __init__(self, window_size: int = 10):
        """Initialize interview-specific emotion analyzer."""
        
        # Interview-focused emotion weights (optimized for 72.72% model)
        self.confidence_weights = {
            'happy': 0.85,      # Strong positive - model is 90% accurate here
            'neutral': 0.75,    # Professional composure - 74% accurate
            'surprise': 0.50,   # Engagement indicator - 86% accurate
            'sad': -0.60,       # Negative indicator
            'fear': -0.70,      # Anxiety indicator - hardest to detect
            'angry': -0.50,     # Frustration indicator
            'disgust': -0.40    # Mild negative
        }
        
        # Confidence level thresholds with detailed feedback
        self.confidence_levels = [
            (0.80, "Excellent", "Outstanding interview presence", "You're projecting confidence and professionalism exceptionally well.", (46, 204, 113)),
            (0.65, "Confident", "Strong professional composure", "Good job! Minor adjustments could make you even more compelling.", (52, 152, 219)),
            (0.50, "Moderate", "Acceptable but room for growth", "Focus on relaxing your facial muscles and maintaining steady eye contact.", (241, 196, 15)),
            (0.35, "Nervous", "Visible signs of anxiety", "Try power posing before your next session. Deep breaths help.", (230, 126, 34)),
            (0.20, "Anxious", "High stress indicators", "Consider practicing with a friend first. You've got this!", (231, 76, 60)),
            (0.00, "Very Anxious", "Significant anxiety detected", "Start with shorter practice sessions. Build confidence gradually.", (192, 57, 43))
        ]
        
        # Rolling window for real-time smoothing (display only)
        self.window_size = window_size
        self.recent_emotions = deque(maxlen=window_size)
        self.recent_confidence = deque(maxlen=window_size)
        
        # FULL SESSION DATA STORAGE - This is the fix!
        self.all_frames: List[FrameData] = []
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Emotion counts for quick stats
        self.emotion_counts = {emotion: 0 for emotion in self.confidence_weights.keys()}
        
        # Segment tracking (for beginning/middle/end analysis)
        self.segment_data = {
            'beginning': [],  # First 20%
            'middle': [],     # Middle 60%
            'end': []         # Last 20%
        }
        
        # Stress spike detection
        self.stress_spikes: List[Tuple[float, float]] = []  # (timestamp, confidence_drop)
        self.last_confidence = 0.5
        
        # Face stability tracking (proxy for eye contact)
        self.face_positions: List[Tuple[float, Tuple[int, int, int, int]]] = []
        self.frames_without_face = 0
        self.total_frames = 0
        
    def calculate_interview_confidence(self, emotions_dict: Dict[str, float]) -> float:
        """Calculate interview-specific confidence score with smart adjustments."""
        
        # Handle smile -> surprise misclassification (model quirk)
        adjusted_emotions = emotions_dict.copy()
        
        if emotions_dict.get('surprise', 0) > 0.4 and emotions_dict.get('happy', 0) > 0.15:
            # Likely a smile being detected as surprise
            surprise_transfer = min(0.35, emotions_dict['surprise'] * 0.5)
            adjusted_emotions['happy'] = min(1.0, adjusted_emotions.get('happy', 0) + surprise_transfer)
            adjusted_emotions['surprise'] = max(0.0, adjusted_emotions.get('surprise', 0) - surprise_transfer)
        
        # Calculate weighted confidence score
        confidence_score = 0.0
        total_weight = 0.0
        
        for emotion, probability in adjusted_emotions.items():
            weight = self.confidence_weights.get(emotion, 0)
            confidence_score += probability * weight
            total_weight += abs(weight) * probability
        
        # Normalize to 0-1 range with better scaling
        normalized_score = (confidence_score + 0.7) / 1.55  # Adjusted for new weights
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return normalized_score
    
    def get_confidence_assessment(self, confidence_score: float) -> Tuple[str, str, str, tuple]:
        """Get confidence level, message, detailed advice, and color."""
        
        for threshold, level, message, advice, color in self.confidence_levels:
            if confidence_score >= threshold:
                return level, message, advice, color
        
        return "Unknown", "Unable to assess", "Check camera and lighting", (128, 128, 128)
    
    def detect_stress_spike(self, current_confidence: float, timestamp: float):
        """Detect sudden drops in confidence (stress spikes)."""
        
        if len(self.recent_confidence) >= 3:
            recent_avg = np.mean(list(self.recent_confidence)[-3:])
            if current_confidence < recent_avg - 0.15:  # 15% drop
                self.stress_spikes.append((timestamp, recent_avg - current_confidence))
        
        self.last_confidence = current_confidence
    
    def add_emotion_data(
        self, 
        emotions_dict: Dict[str, float],
        face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """Add new emotion data and return real-time analysis."""
        
        timestamp = time.time() - self.session_start
        self.total_frames += 1
        
        # Calculate confidence
        confidence_score = self.calculate_interview_confidence(emotions_dict)
        
        # Detect stress spikes
        self.detect_stress_spike(confidence_score, timestamp)
        
        # Track face detection
        face_detected = face_box is not None
        if not face_detected:
            self.frames_without_face += 1
        else:
            self.face_positions.append((timestamp, face_box))
        
        # Get dominant emotion
        dominant_emotion = max(emotions_dict, key=emotions_dict.get)
        self.emotion_counts[dominant_emotion] += 1
        
        # Create frame data
        frame_data = FrameData(
            timestamp=timestamp,
            emotions=emotions_dict.copy(),
            confidence_score=confidence_score,
            dominant_emotion=dominant_emotion,
            face_detected=face_detected,
            face_position=face_box
        )
        
        # Store ALL frame data (THE FIX!)
        self.all_frames.append(frame_data)
        
        # Update rolling windows for real-time display
        self.recent_emotions.append(emotions_dict)
        self.recent_confidence.append(confidence_score)
        
        # Update segment data
        self._update_segment_data(frame_data)
        
        # Calculate smoothed values for display
        smoothed_emotions = self._calculate_smoothed_emotions()
        smoothed_confidence = np.mean(list(self.recent_confidence)) if self.recent_confidence else confidence_score
        
        # Get assessment
        level, message, advice, color = self.get_confidence_assessment(smoothed_confidence)
        
        # Calculate stability
        stability = self._calculate_stability()
        
        return {
            'current_confidence': confidence_score,
            'smoothed_confidence': smoothed_confidence,
            'confidence_level': level,
            'message': message,
            'advice': advice,
            'color': color,
            'stability': stability,
            'dominant_emotion': dominant_emotion,
            'smoothed_emotions': smoothed_emotions,
            'session_duration': timestamp,
            'frames_analyzed': len(self.all_frames),
            'face_detected': face_detected
        }
    
    def _update_segment_data(self, frame_data: FrameData):
        """Assign frame to appropriate segment for later analysis."""
        # We'll finalize segments when generating the report
        # For now, just store the data
        pass
    
    def _calculate_smoothed_emotions(self) -> Dict[str, float]:
        """Calculate smoothed emotion values over recent window."""
        
        if not self.recent_emotions:
            return {}
        
        smoothed = {}
        for emotion in self.confidence_weights.keys():
            values = [frame.get(emotion, 0) for frame in self.recent_emotions]
            smoothed[emotion] = float(np.mean(values))
        
        return smoothed
    
    def _calculate_stability(self) -> Dict[str, float]:
        """Calculate emotional stability metrics."""
        
        if len(self.recent_confidence) < 3:
            return {'variance': 0.0, 'trend': 'stable', 'consistency': 1.0}
        
        confidence_list = list(self.recent_confidence)
        
        # Confidence variance
        confidence_variance = float(np.var(confidence_list))
        
        # Trend calculation
        recent_avg = np.mean(confidence_list[-3:])
        earlier_avg = np.mean(confidence_list[:-3]) if len(confidence_list) > 3 else recent_avg
        
        if recent_avg > earlier_avg + 0.05:
            trend = 'improving'
        elif recent_avg < earlier_avg - 0.05:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Consistency score
        consistency = max(0.0, 1.0 - (confidence_variance * 10))
        
        return {
            'variance': confidence_variance,
            'trend': trend,
            'consistency': consistency
        }
    
    def calculate_engagement_score(self) -> float:
        """Calculate overall engagement based on emotion variety and positivity."""
        
        if not self.all_frames:
            return 0.0
        
        # Factors for engagement:
        # 1. Presence of positive emotions (happy, surprise)
        # 2. Low presence of negative emotions
        # 3. Face detection rate
        # 4. Emotional expressiveness (not just neutral)
        
        total_frames = len(self.all_frames)
        
        # Positive emotion ratio
        positive_frames = sum(1 for f in self.all_frames 
                            if f.dominant_emotion in ['happy', 'surprise'])
        positive_ratio = positive_frames / total_frames
        
        # Negative emotion ratio
        negative_frames = sum(1 for f in self.all_frames 
                            if f.dominant_emotion in ['sad', 'fear', 'angry', 'disgust'])
        negative_ratio = negative_frames / total_frames
        
        # Face detection rate
        face_rate = 1.0 - (self.frames_without_face / total_frames)
        
        # Expressiveness (not always neutral)
        neutral_frames = sum(1 for f in self.all_frames if f.dominant_emotion == 'neutral')
        expressiveness = 1.0 - (neutral_frames / total_frames) * 0.5  # Some neutral is fine
        
        # Weighted engagement score
        engagement = (
            positive_ratio * 0.35 +
            (1 - negative_ratio) * 0.25 +
            face_rate * 0.25 +
            expressiveness * 0.15
        )
        
        return min(1.0, max(0.0, engagement))
    
    def calculate_face_stability(self) -> float:
        """Calculate face position stability (proxy for eye contact/focus)."""
        
        if len(self.face_positions) < 2:
            return 1.0
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(self.face_positions)):
            _, (x1, y1, w1, h1) = self.face_positions[i-1]
            _, (x2, y2, w2, h2) = self.face_positions[i]
            
            # Center point movement
            cx1, cy1 = x1 + w1/2, y1 + h1/2
            cx2, cy2 = x2 + w2/2, y2 + h2/2
            
            movement = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
            movements.append(movement)
        
        if not movements:
            return 1.0
        
        # Normalize movement (assuming 640x480 frame)
        avg_movement = np.mean(movements)
        max_expected_movement = 50  # pixels
        
        stability = 1.0 - min(1.0, avg_movement / max_expected_movement)
        return stability
    
    def get_segment_analysis(self) -> Dict[str, Dict]:
        """Analyze performance by interview segments."""
        
        if not self.all_frames:
            return {}
        
        total_frames = len(self.all_frames)
        
        # Define segments
        beginning_end = int(total_frames * 0.2)
        middle_start = beginning_end
        middle_end = int(total_frames * 0.8)
        
        segments = {
            'beginning': self.all_frames[:beginning_end],
            'middle': self.all_frames[middle_start:middle_end],
            'end': self.all_frames[middle_end:]
        }
        
        analysis = {}
        for segment_name, frames in segments.items():
            if not frames:
                continue
                
            confidences = [f.confidence_score for f in frames]
            emotions = [f.dominant_emotion for f in frames]
            
            # Most common emotion in segment
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'unknown'
            
            analysis[segment_name] = {
                'avg_confidence': float(np.mean(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'dominant_emotion': dominant,
                'frame_count': len(frames),
                'confidence_trend': 'improving' if confidences[-1] > confidences[0] else 'declining' if confidences[-1] < confidences[0] else 'stable'
            }
        
        return analysis
    
    def get_improvement_areas(self) -> List[Dict[str, str]]:
        """Generate specific improvement recommendations based on data."""
        
        improvements = []
        
        if not self.all_frames:
            return [{'area': 'No Data', 'recommendation': 'Complete an interview session first.'}]
        
        # Analyze the data
        avg_confidence = np.mean([f.confidence_score for f in self.all_frames])
        face_rate = 1.0 - (self.frames_without_face / len(self.all_frames))
        
        # Check for specific issues
        
        # 1. Low overall confidence
        if avg_confidence < 0.5:
            improvements.append({
                'area': 'Overall Confidence',
                'priority': 'high',
                'recommendation': 'Practice power poses before interviews. Stand tall with hands on hips for 2 minutes.',
                'metric': f'{avg_confidence:.0%} average confidence'
            })
        
        # 2. Too much fear/anxiety
        fear_ratio = self.emotion_counts.get('fear', 0) / len(self.all_frames)
        if fear_ratio > 0.1:
            improvements.append({
                'area': 'Anxiety Management',
                'priority': 'high',
                'recommendation': 'Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat 3 times.',
                'metric': f'{fear_ratio:.0%} frames showing anxiety'
            })
        
        # 3. Not enough positive expression
        happy_ratio = self.emotion_counts.get('happy', 0) / len(self.all_frames)
        if happy_ratio < 0.15:
            improvements.append({
                'area': 'Positive Expression',
                'priority': 'medium',
                'recommendation': 'Practice your "interview smile" - slight, genuine, and warm. Not too wide.',
                'metric': f'Only {happy_ratio:.0%} frames with positive expression'
            })
        
        # 4. Too much neutral (might seem disengaged)
        neutral_ratio = self.emotion_counts.get('neutral', 0) / len(self.all_frames)
        if neutral_ratio > 0.7:
            improvements.append({
                'area': 'Engagement',
                'priority': 'medium',
                'recommendation': 'Show more enthusiasm when discussing your achievements. Vary your expressions.',
                'metric': f'{neutral_ratio:.0%} frames appear neutral'
            })
        
        # 5. Poor face detection (looking away)
        if face_rate < 0.8:
            improvements.append({
                'area': 'Eye Contact',
                'priority': 'high',
                'recommendation': 'Maintain eye contact with the camera. Place a sticker near the lens as a reminder.',
                'metric': f'Face visible only {face_rate:.0%} of the time'
            })
        
        # 6. Stress spikes
        if len(self.stress_spikes) > 3:
            improvements.append({
                'area': 'Stress Management',
                'priority': 'medium',
                'recommendation': 'You had sudden confidence drops. Prepare answers for tough questions in advance.',
                'metric': f'{len(self.stress_spikes)} stress spikes detected'
            })
        
        # 7. Segment-specific issues
        segments = self.get_segment_analysis()
        if segments:
            beginning = segments.get('beginning', {})
            end = segments.get('end', {})
            
            if beginning.get('avg_confidence', 1) < 0.4:
                improvements.append({
                    'area': 'First Impressions',
                    'priority': 'high',
                    'recommendation': 'Your opening is weak. Prepare and practice your introduction until it feels natural.',
                    'metric': f'{beginning.get("avg_confidence", 0):.0%} confidence at start'
                })
            
            if end.get('avg_confidence', 1) < beginning.get('avg_confidence', 0):
                improvements.append({
                    'area': 'Stamina',
                    'priority': 'medium',
                    'recommendation': 'Your confidence dropped toward the end. Practice longer sessions to build endurance.',
                    'metric': 'Declining confidence trend'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        improvements.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        if not improvements:
            improvements.append({
                'area': 'Great Job!',
                'priority': 'low',
                'recommendation': 'Your interview presence is strong. Keep practicing to maintain this level.',
                'metric': f'{avg_confidence:.0%} average confidence'
            })
        
        return improvements
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary with all metrics."""
        
        if not self.all_frames:
            return {'error': 'No data collected'}
        
        confidence_scores = [f.confidence_score for f in self.all_frames]
        timestamps = [f.timestamp for f in self.all_frames]
        
        # Most common emotion
        most_common_emotion = max(self.emotion_counts, key=self.emotion_counts.get)
        
        # Confidence statistics
        avg_confidence = float(np.mean(confidence_scores))
        min_confidence = float(np.min(confidence_scores))
        max_confidence = float(np.max(confidence_scores))
        std_confidence = float(np.std(confidence_scores))
        
        # Session duration
        session_duration = self.all_frames[-1].timestamp if self.all_frames else 0
        
        # Get assessment
        level, message, advice, color = self.get_confidence_assessment(avg_confidence)
        
        return {
            'session_id': self.session_id,
            'session_duration': session_duration,
            'frames_analyzed': len(self.all_frames),
            'fps': len(self.all_frames) / session_duration if session_duration > 0 else 0,
            
            # Confidence metrics
            'average_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'std_confidence': std_confidence,
            
            # Emotion data
            'most_common_emotion': most_common_emotion,
            'emotion_distribution': dict(self.emotion_counts),
            
            # Assessment
            'final_assessment': (level, message, advice),
            'assessment_color': color,
            
            # Advanced metrics
            'engagement_score': self.calculate_engagement_score(),
            'face_stability': self.calculate_face_stability(),
            'face_detection_rate': 1.0 - (self.frames_without_face / len(self.all_frames)),
            'stress_spikes_count': len(self.stress_spikes),
            
            # Segment analysis
            'segment_analysis': self.get_segment_analysis(),
            
            # Improvement areas
            'improvement_areas': self.get_improvement_areas(),
            
            # Raw time-series data for plotting (THE KEY FIX!)
            'time_series': {
                'timestamps': timestamps,
                'confidence_scores': confidence_scores,
                'emotions_over_time': [
                    {
                        'timestamp': f.timestamp,
                        'emotions': f.emotions,
                        'dominant': f.dominant_emotion
                    }
                    for f in self.all_frames
                ]
            },
            
            # Stress spike data
            'stress_spikes': self.stress_spikes
        }
    
    def save_session_data(self, filename: str = None) -> str:
        """Save complete session data to JSON file."""
        
        if filename is None:
            os.makedirs("outputs", exist_ok=True)
            filename = f"outputs/interview_session_{self.session_id}.json"
        
        summary = self.get_session_summary()
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Session data saved to: {filename}")
        return filename
    
    def reset_session(self):
        """Reset for a new session."""
        self.all_frames = []
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.emotion_counts = {emotion: 0 for emotion in self.confidence_weights.keys()}
        self.stress_spikes = []
        self.face_positions = []
        self.frames_without_face = 0
        self.total_frames = 0
        self.recent_emotions.clear()
        self.recent_confidence.clear()


# Test the analyzer
if __name__ == "__main__":
    import random
    
    print("Testing Enhanced Interview Emotion Analyzer")
    print("=" * 50)
    
    analyzer = InterviewEmotionAnalyzer()
    
    # Simulate a 30-second interview with varying emotions
    for i in range(90):  # 3 FPS for 30 seconds
        # Simulate different phases
        if i < 20:  # Nervous start
            emotions = {
                'neutral': 0.4 + random.uniform(-0.1, 0.1),
                'fear': 0.3 + random.uniform(-0.1, 0.1),
                'happy': 0.1 + random.uniform(-0.05, 0.05),
                'surprise': 0.1,
                'sad': 0.05,
                'angry': 0.03,
                'disgust': 0.02
            }
        elif i < 60:  # Warming up
            emotions = {
                'neutral': 0.5 + random.uniform(-0.1, 0.1),
                'happy': 0.25 + random.uniform(-0.1, 0.1),
                'surprise': 0.1,
                'fear': 0.08,
                'sad': 0.04,
                'angry': 0.02,
                'disgust': 0.01
            }
        else:  # Confident ending
            emotions = {
                'neutral': 0.4 + random.uniform(-0.1, 0.1),
                'happy': 0.4 + random.uniform(-0.1, 0.1),
                'surprise': 0.1,
                'fear': 0.05,
                'sad': 0.03,
                'angry': 0.01,
                'disgust': 0.01
            }
        
        # Simulate face box
        face_box = (100 + random.randint(-5, 5), 100 + random.randint(-5, 5), 200, 200)
        
        analysis = analyzer.add_emotion_data(emotions, face_box)
        
        if i % 30 == 0:
            print(f"\nFrame {i}: {analysis['confidence_level']} ({analysis['smoothed_confidence']:.2f})")
    
    # Get summary
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    
    summary = analyzer.get_session_summary()
    print(f"Duration: {summary['session_duration']:.1f}s")
    print(f"Frames: {summary['frames_analyzed']}")
    print(f"Average Confidence: {summary['average_confidence']:.2%}")
    print(f"Engagement Score: {summary['engagement_score']:.2%}")
    print(f"Face Stability: {summary['face_stability']:.2%}")
    
    print("\nSegment Analysis:")
    for segment, data in summary['segment_analysis'].items():
        print(f"  {segment.title()}: {data['avg_confidence']:.2%} ({data['confidence_trend']})")
    
    print("\nImprovement Areas:")
    for area in summary['improvement_areas'][:3]:
        print(f"  [{area['priority'].upper()}] {area['area']}: {area['recommendation']}")
    
    # Save session
    analyzer.save_session_data()
