# src/interview_report_generator.py
"""
INSIGHT-FIRST Interview Performance Report

Design Philosophy:
- The data should SCREAM the insight at first glance
- No data vomit - only what matters
- Hero metric front and center
- One clear action item
- Supporting data only if needed

Color Psychology (CONSISTENT):
- Red (0-59%): NEEDS PRACTICE
- Yellow/Amber (60-79%): DEVELOPING  
- Green (80-100%): EXCELLENT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class InterviewReportGenerator:
    """Generate insight-first interview performance reports."""
    
    # Typical improvement benchmark (for context)
    TYPICAL_IMPROVEMENT = 0.12  # 12% is average improvement
    
    def __init__(self):
        # CONSISTENT color scale - same rules for ALL metrics
        self.colors = {
            # Performance colors (UNIVERSAL scale)
            'excellent': '#10B981',    # Green (80-100%)
            'good': '#22C55E',         # Lighter green (75-79%)
            'developing': '#F59E0B',   # Amber/Yellow (60-74%)
            'needs_work': '#EF4444',   # Red (0-59%)
            
            # Chart gradient colors
            'chart_start': '#FFF4E6',  # Light orange (beginning)
            'chart_end': '#E8F5E9',    # Light green (end/growth)
            'chart_line': '#F59E0B',   # Amber line
            
            # Neutrals
            'dark': '#1F2937',         # Almost black
            'medium': '#6B7280',       # Gray
            'light': '#F3F4F6',        # Light gray
            'white': '#FFFFFF',
            
            # Divider
            'divider': '#E5E7EB',      # Subtle gray
            
            # Focus indicator
            'focus_bg': '#FEF3C7',     # Light amber background
        }
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score - UNIVERSAL scale for all metrics."""
        if score >= 0.80:
            return self.colors['excellent']
        elif score >= 0.75:
            return self.colors['good']
        elif score >= 0.60:
            return self.colors['developing']
        else:
            return self.colors['needs_work']
    
    def _get_score_label(self, score: float) -> str:
        """Get human-readable label for score."""
        if score >= 0.85:
            return "EXCELLENT"
        elif score >= 0.75:
            return "GOOD"
        elif score >= 0.60:
            return "DEVELOPING"
        else:
            return "NEEDS PRACTICE"
    
    def _get_improvement_context(self, growth: float) -> str:
        """Get context for how the improvement compares to typical."""
        if growth >= self.TYPICAL_IMPROVEMENT * 2:
            return f"That's 2x the typical improvement!"
        elif growth >= self.TYPICAL_IMPROVEMENT * 1.5:
            return f"Above average (typical is ~{self.TYPICAL_IMPROVEMENT:.0%})"
        elif growth >= self.TYPICAL_IMPROVEMENT:
            return "Right on track with typical progress"
        elif growth > 0:
            return "Some improvement - keep practicing!"
        else:
            return ""
    
    def _get_main_insight(self, data: Dict) -> Dict:
        """Extract THE main insight from the data."""
        
        avg_conf = data.get('average_confidence', 0.5)
        segments = data.get('segment_analysis', {})
        improvements = data.get('improvement_areas', [])
        stress_count = data.get('stress_spikes_count', 0)
        
        beginning = segments.get('beginning', {}).get('avg_confidence', avg_conf)
        end = segments.get('end', {}).get('avg_confidence', avg_conf)
        growth = end - beginning
        growth_context = self._get_improvement_context(growth)
        
        # Check for major issues first
        if beginning < 0.4:
            return {
                'headline': "WEAK START",
                'subtext': "Your first impression needs work",
                'context': f"Started at {beginning:.0%}. First 30 seconds are crucial.",
                'action': "Practice your opening until it's automatic.",
                'color': self.colors['needs_work']
            }
        
        if end < beginning - 0.15:
            return {
                'headline': "FADING FINISH", 
                'subtext': "You lost momentum at the end",
                'context': f"Dropped from {beginning:.0%} to {end:.0%}. Strong closings win offers.",
                'action': "Prepare a confident closing statement.",
                'color': self.colors['needs_work']
            }
        
        if stress_count > 3:
            return {
                'headline': "STRESS VISIBLE",
                'subtext': f"{stress_count} anxiety spikes detected",
                'context': "Interviewers notice sudden confidence drops.",
                'action': "Practice tough questions until they don't faze you.",
                'color': self.colors['needs_work']
            }
        
        if avg_conf < 0.45:
            return {
                'headline': "BUILD CONFIDENCE",
                'subtext': "You appear nervous on camera",
                'context': f"Overall: {avg_conf:.0%}. Body language matters.",
                'action': "Try power posing for 2 min before your next session.",
                'color': self.colors['needs_work']
            }
        
        # Positive insights
        if growth >= 0.15:
            return {
                'headline': "STRONG GROWTH",
                'subtext': f"+{growth:.0%} improvement during session",
                'context': growth_context,
                'action': "Work on starting stronger to match your finish.",
                'color': self.colors['excellent']
            }
        
        if growth >= 0.08:
            return {
                'headline': "BUILDING MOMENTUM",
                'subtext': f"+{growth:.0%} improvement",
                'context': growth_context,
                'action': "Practice starting at your peak level.",
                'color': self.colors['developing']
            }
        
        if avg_conf >= 0.80:
            return {
                'headline': "INTERVIEW READY",
                'subtext': "You project strong confidence",
                'context': f"Consistent {avg_conf:.0%} throughout. Excellent!",
                'action': "You're prepared. Trust yourself.",
                'color': self.colors['excellent']
            }
        
        if avg_conf >= 0.65:
            return {
                'headline': "SOLID PERFORMANCE",
                'subtext': "Good foundation, room to grow",
                'context': f"Steady at {avg_conf:.0%}. A bit more practice will help.",
                'action': improvements[0]['recommendation'] if improvements else "Keep practicing!",
                'color': self.colors['developing']
            }
        
        # Default
        return {
            'headline': "KEEP PRACTICING",
            'subtext': "You're making progress",
            'context': f"Currently at {avg_conf:.0%}. Each session helps.",
            'action': "Focus on one thing at a time.",
            'color': self.colors['developing']
        }
    
    def _find_weakest_metric(self, metrics: List[Tuple[str, float]]) -> int:
        """Find index of weakest metric to highlight."""
        min_val = min(m[1] for m in metrics)
        for i, (_, val) in enumerate(metrics):
            if val == min_val:
                return i
        return 0
    
    def generate_visual_report(
        self, 
        session_data: Dict, 
        save_path: str = "outputs/"
    ) -> str:
        """Generate insight-first visual report."""
        
        # Load data
        if isinstance(session_data, str):
            with open(session_data, 'r') as f:
                data = json.load(f)
        else:
            data = session_data
        
        # Get the main insight
        insight = self._get_main_insight(data)
        avg_conf = data.get('average_confidence', 0.5)
        score_color = self._get_score_color(avg_conf)
        
        # Get segment values
        segments = data.get('segment_analysis', {})
        start_value = segments.get('beginning', {}).get('avg_confidence', avg_conf)
        middle_value = segments.get('middle', {}).get('avg_confidence', avg_conf)
        end_value = segments.get('end', {}).get('avg_confidence', avg_conf)
        
        # Create clean figure
        fig = plt.figure(figsize=(12, 18), facecolor=self.colors['white'])
        
        # === SECTION 1: HERO SECTION ===
        
        ax_hero = fig.add_axes([0.05, 0.60, 0.9, 0.36])
        ax_hero.set_xlim(0, 1)
        ax_hero.set_ylim(0, 1)
        ax_hero.axis('off')
        
        # Background
        ax_hero.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=self.colors['light'], transform=ax_hero.transAxes
        ))
        
        # Giant score circle
        circle = mpatches.Circle(
            (0.5, 0.68), 0.22, 
            facecolor=score_color,
            edgecolor=self.colors['white'],
            linewidth=8,
            transform=ax_hero.transAxes
        )
        ax_hero.add_patch(circle)
        
        # Score percentage - THE NUMBER
        ax_hero.text(0.5, 0.71, f"{avg_conf:.0%}", 
                    fontsize=68, fontweight='bold', color='white',
                    ha='center', va='center', transform=ax_hero.transAxes)
        
        # Score label - CLEARLY explains what this measures
        ax_hero.text(0.5, 0.56, "AVERAGE CONFIDENCE",
                    fontsize=11, color='white', fontweight='bold',
                    ha='center', va='center', transform=ax_hero.transAxes,
                    alpha=0.95)
        
        # Tiny context below circle
        ax_hero.text(0.5, 0.40, "Based on your entire session",
                    fontsize=9, color=self.colors['medium'], style='italic',
                    ha='center', va='center', transform=ax_hero.transAxes)
        
        # Status label - BOLDER
        ax_hero.text(0.5, 0.32, self._get_score_label(avg_conf),
                    fontsize=18, fontweight='black', color=score_color,
                    ha='center', va='center', transform=ax_hero.transAxes)
        
        # Main headline - THE INSIGHT
        ax_hero.text(0.5, 0.18, insight['headline'],
                    fontsize=34, fontweight='bold', color=self.colors['dark'],
                    ha='center', va='center', transform=ax_hero.transAxes)
        
        # Subtext
        ax_hero.text(0.5, 0.09, insight['subtext'],
                    fontsize=17, color=self.colors['dark'], fontweight='medium',
                    ha='center', va='center', transform=ax_hero.transAxes)
        
        # Context line (the "2x typical" etc.)
        if insight.get('context'):
            ax_hero.text(0.5, 0.02, insight['context'],
                        fontsize=12, color=self.colors['excellent'], fontweight='bold',
                        ha='center', va='center', transform=ax_hero.transAxes)
        
        # === SECTION 2: THE ONE ACTION ===
        
        ax_action = fig.add_axes([0.08, 0.51, 0.84, 0.07])
        ax_action.set_xlim(0, 1)
        ax_action.set_ylim(0, 1)
        ax_action.axis('off')
        
        # Action box
        ax_action.add_patch(mpatches.FancyBboxPatch(
            (0, 0.1), 1, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=self.colors['dark'],
            transform=ax_action.transAxes
        ))
        
        ax_action.text(0.04, 0.5, "NEXT STEP",
                      fontsize=11, color=self.colors['excellent'], fontweight='bold',
                      ha='left', va='center', transform=ax_action.transAxes)
        
        ax_action.text(0.18, 0.5, insight['action'],
                      fontsize=15, color='white', fontweight='medium',
                      ha='left', va='center', transform=ax_action.transAxes)
        
        # === SECTION 3: JOURNEY VISUALIZATION ===
        
        # Title ABOVE the chart - BIGGER
        fig.text(0.5, 0.49, "YOUR INTERVIEW JOURNEY", 
                fontsize=14, fontweight='bold', color=self.colors['dark'], ha='center')
        
        ax_journey = fig.add_axes([0.10, 0.28, 0.80, 0.18])
        ax_journey.set_facecolor(self.colors['white'])
        
        # Get time series data
        time_series = data.get('time_series', {})
        timestamps = time_series.get('timestamps', [])
        confidence_scores = time_series.get('confidence_scores', [])
        
        if len(timestamps) > 1:
            # Smooth the line
            if len(confidence_scores) >= 10:
                window = len(confidence_scores) // 10
                smooth = np.convolve(confidence_scores, np.ones(window)/window, mode='valid')
                smooth_t = timestamps[window-1:]
            else:
                smooth = list(confidence_scores)
                smooth_t = list(timestamps)
            
            max_t = max(smooth_t)
            
            # === GRADIENT FILL (orange to green) ===
            # Create gradient by filling small segments with interpolated colors
            n_segments = 50
            for i in range(n_segments):
                t_start = i / n_segments
                t_end = (i + 1) / n_segments
                
                # Interpolate color from orange to green
                r = int(255 * (1 - t_end) + 232 * t_end)
                g = int(244 * (1 - t_end) + 245 * t_end)
                b = int(230 * (1 - t_end) + 233 * t_end)
                color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Find indices in this time range
                idx_start = int(t_start * len(smooth_t))
                idx_end = min(int(t_end * len(smooth_t)) + 1, len(smooth_t))
                
                if idx_end > idx_start:
                    segment_t = smooth_t[idx_start:idx_end]
                    segment_v = smooth[idx_start:idx_end]
                    ax_journey.fill_between(segment_t, segment_v, alpha=0.6, color=color, linewidth=0)
            
            # Main line
            ax_journey.plot(smooth_t, smooth, linewidth=4, color=self.colors['chart_line'], zorder=3)
            
            # === THREE MARKERS: START, MIDDLE, END ===
            start_color = self._get_score_color(start_value)
            middle_color = self._get_score_color(middle_value)
            end_color = self._get_score_color(end_value)
            
            mid_t = smooth_t[len(smooth_t)//2]
            
            # Markers
            ax_journey.scatter([smooth_t[0]], [start_value], s=150, color=start_color, 
                              zorder=5, edgecolor='white', linewidth=3)
            ax_journey.scatter([mid_t], [middle_value], s=120, color=middle_color, 
                              zorder=5, edgecolor='white', linewidth=2)
            ax_journey.scatter([smooth_t[-1]], [end_value], s=150, color=end_color, 
                              zorder=5, edgecolor='white', linewidth=3)
            
            # Labels above markers
            ax_journey.text(smooth_t[0], start_value + 0.12, f"{start_value:.0%}",
                           fontsize=14, ha='center', color=start_color, fontweight='bold')
            ax_journey.text(mid_t, middle_value + 0.12, f"{middle_value:.0%}",
                           fontsize=12, ha='center', color=middle_color, fontweight='bold')
            ax_journey.text(smooth_t[-1], end_value + 0.12, f"{end_value:.0%}",
                           fontsize=14, ha='center', color=end_color, fontweight='bold')
        
        ax_journey.set_xlim(0, max(timestamps) if timestamps else 1)
        ax_journey.set_ylim(0, 1.20)
        ax_journey.spines['top'].set_visible(False)
        ax_journey.spines['right'].set_visible(False)
        ax_journey.spines['left'].set_visible(False)
        ax_journey.spines['bottom'].set_color(self.colors['divider'])
        ax_journey.tick_params(left=False, labelleft=False)
        # Hide x-axis tick labels to avoid overlap with our custom labels
        ax_journey.tick_params(bottom=False, labelbottom=False)
        
        # Phase labels below x-axis - clean positioning, no overlap with ticks
        if timestamps:
            max_t = max(timestamps)
            # Use figure text instead of axes text to avoid tick overlap
            ax_journey.text(max_t * 0.02, -0.06, "START", fontsize=12, fontweight='bold',
                           color=self.colors['dark'], ha='left')
            ax_journey.text(max_t * 0.50, -0.06, "MIDDLE", fontsize=12, fontweight='bold',
                           color=self.colors['medium'], ha='center')
            ax_journey.text(max_t * 0.98, -0.06, "END", fontsize=12, fontweight='bold',
                           color=self.colors['dark'], ha='right')
        
        # === DIVIDER LINE ===
        ax_divider = fig.add_axes([0.10, 0.22, 0.80, 0.01])
        ax_divider.axhline(y=0.5, color=self.colors['divider'], linewidth=2)
        ax_divider.axis('off')
        
        # === SECTION 4: SUPPORTING METRICS ===
        
        # Section header - BOLDER
        fig.text(0.5, 0.20, "SESSION QUALITY METRICS", 
                fontsize=12, color=self.colors['dark'], ha='center', fontweight='bold')
        
        ax_stats = fig.add_axes([0.05, 0.06, 0.90, 0.12])
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        # Metrics - user-friendly labels (not technical terms)
        metrics = [
            ("EXPRESSIVENESS", data.get('engagement_score', 0)),
            ("CAMERA PRESENCE", data.get('face_detection_rate', 0)),  # Not "Face Detection" - that's technical
            ("COMPOSURE", data.get('face_stability', 0)),  # "Composure" > "Head Stability"
        ]
        
        weakest_idx = self._find_weakest_metric(metrics)
        
        for i, (label, value) in enumerate(metrics):
            x = 0.17 + i * 0.33
            
            color = self._get_score_color(value)
            
            if i == weakest_idx:
                # === IMPROVED FOCUS INDICATOR ===
                # Background highlight box
                ax_stats.add_patch(mpatches.FancyBboxPatch(
                    (x - 0.12, 0.15), 0.24, 0.75,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor=self.colors['focus_bg'],
                    edgecolor=self.colors['developing'],
                    linewidth=2,
                    transform=ax_stats.transAxes
                ))
                fontsize = 34
                # "Work on this" indicator
                ax_stats.text(x, 0.95, "Work on this",
                             fontsize=9, color=self.colors['developing'], fontweight='bold',
                             ha='center', va='center', transform=ax_stats.transAxes)
            else:
                fontsize = 28
            
            # Value
            ax_stats.text(x, 0.58, f"{value:.0%}", fontsize=fontsize, fontweight='bold',
                         color=color, ha='center', va='center', transform=ax_stats.transAxes)
            
            # Label
            ax_stats.text(x, 0.25, label, fontsize=10, 
                         color=self.colors['dark'] if i == weakest_idx else self.colors['medium'],
                         ha='center', va='center', transform=ax_stats.transAxes,
                         fontweight='bold' if i == weakest_idx else 'normal')
        
        # === FOOTER ===
        
        session_id = data.get('session_id', '')
        duration = data.get('session_duration', 0)
        frames = data.get('frames_analyzed', 0)
        
        fig.text(0.5, 0.02, f"Session {session_id}  |  {duration:.0f}s  |  {frames} frames analyzed",
                fontsize=10, color=self.colors['medium'], ha='center')
        
        # Save
        os.makedirs(save_path, exist_ok=True)
        session_id = data.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"{save_path}interview_report_{session_id}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight', 
                   facecolor=self.colors['white'], edgecolor='none')
        
        print(f"Report saved: {filename}")
        
        plt.close()
        return filename
    
    def generate_text_report(self, session_data: Dict, save_path: str = "outputs/") -> str:
        """Generate comprehensive detailed text report with all metrics."""
        
        if isinstance(session_data, str):
            with open(session_data, 'r') as f:
                data = json.load(f)
        else:
            data = session_data
        
        insight = self._get_main_insight(data)
        avg_conf = data.get('average_confidence', 0.5)
        
        # Get all data
        segments = data.get('segment_analysis', {})
        start_value = segments.get('beginning', {}).get('avg_confidence', avg_conf)
        middle_value = segments.get('middle', {}).get('avg_confidence', avg_conf)
        end_value = segments.get('end', {}).get('avg_confidence', avg_conf)
        
        engagement = data.get('engagement_score', 0)
        face_detection = data.get('face_detection_rate', 0)
        stability = data.get('face_stability', 0)
        stress_spikes = data.get('stress_spikes_count', 0)
        
        session_id = data.get('session_id', 'N/A')
        duration = data.get('session_duration', 0)
        frames = data.get('frames_analyzed', 0)
        
        time_series = data.get('time_series', {})
        confidence_scores = time_series.get('confidence_scores', [])
        
        # Calculate additional stats
        if confidence_scores:
            min_conf = min(confidence_scores)
            max_conf = max(confidence_scores)
            std_conf = float(np.std(confidence_scores))
            
            # Find best and worst moments
            best_idx = confidence_scores.index(max_conf)
            worst_idx = confidence_scores.index(min_conf)
            best_time = best_idx * (duration / len(confidence_scores)) if len(confidence_scores) > 0 else 0
            worst_time = worst_idx * (duration / len(confidence_scores)) if len(confidence_scores) > 0 else 0
        else:
            min_conf = max_conf = avg_conf
            std_conf = 0
            best_time = worst_time = 0
        
        growth = end_value - start_value
        growth_context = self._get_improvement_context(growth)
        
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("                    INTERVIEW PERFORMANCE REPORT")
        lines.append("                         DETAILED ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        # Session Info
        lines.append("-" * 70)
        lines.append("SESSION INFORMATION")
        lines.append("-" * 70)
        lines.append(f"  Session ID:        {session_id}")
        lines.append(f"  Duration:          {duration:.1f} seconds")
        lines.append(f"  Frames Analyzed:   {frames}")
        lines.append(f"  Frame Rate:        {frames/duration:.1f} FPS" if duration > 0 else "  Frame Rate:        N/A")
        lines.append("")
        
        # Executive Summary
        lines.append("-" * 70)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 70)
        lines.append(f"  Overall Score:     {avg_conf:.1%} ({self._get_score_label(avg_conf)})")
        lines.append(f"  Headline:          {insight['headline']}")
        lines.append(f"  Summary:           {insight['subtext']}")
        if insight.get('context'):
            lines.append(f"  Context:           {insight['context']}")
        lines.append("")
        lines.append(f"  RECOMMENDED ACTION:")
        lines.append(f"  >>> {insight['action']}")
        lines.append("")
        
        # Confidence Analysis
        lines.append("-" * 70)
        lines.append("CONFIDENCE ANALYSIS")
        lines.append("-" * 70)
        lines.append("")
        lines.append("  Overall Statistics:")
        lines.append(f"    Average:         {avg_conf:.1%}")
        lines.append(f"    Minimum:         {min_conf:.1%}")
        lines.append(f"    Maximum:         {max_conf:.1%}")
        lines.append(f"    Std Deviation:   {std_conf:.3f} ({self._interpret_std(std_conf)})")
        lines.append("")
        lines.append("  Segment Breakdown:")
        lines.append(f"    Beginning:       {start_value:.1%} {self._get_trend_arrow(0, start_value)}")
        lines.append(f"    Middle:          {middle_value:.1%} {self._get_trend_arrow(start_value, middle_value)}")
        lines.append(f"    End:             {end_value:.1%} {self._get_trend_arrow(middle_value, end_value)}")
        lines.append("")
        lines.append("  Growth Analysis:")
        lines.append(f"    Total Change:    {'+' if growth >= 0 else ''}{growth:.1%}")
        lines.append(f"    Interpretation:  {growth_context}")
        lines.append(f"    Typical Range:   10-15% improvement is average")
        lines.append("")
        lines.append("  Key Moments:")
        lines.append(f"    Peak Confidence: {max_conf:.1%} at {best_time:.1f}s")
        lines.append(f"    Low Point:       {min_conf:.1%} at {worst_time:.1f}s")
        lines.append(f"    Stress Spikes:   {stress_spikes} detected")
        lines.append("")
        
        # Session Quality Metrics
        lines.append("-" * 70)
        lines.append("SESSION QUALITY METRICS")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  Expressiveness:    {engagement:.1%} ({self._get_score_label(engagement)})")
        lines.append(f"    > Measures how varied your facial expressions were")
        lines.append(f"    > Higher = more engaging, dynamic presence")
        if engagement < 0.75:
            lines.append(f"    ! TIP: Try to show more natural reactions and expressions")
        lines.append("")
        lines.append(f"  Face Detection:    {face_detection:.1%} ({self._get_score_label(face_detection)})")
        lines.append(f"    > Percentage of frames where your face was visible")
        lines.append(f"    > Target: 95%+ for reliable analysis")
        if face_detection < 0.90:
            lines.append(f"    ! TIP: Ensure good lighting and face the camera directly")
        lines.append("")
        lines.append(f"  Head Stability:    {stability:.1%} ({self._get_score_label(stability)})")
        lines.append(f"    > How steady your head position was during the session")
        lines.append(f"    > Higher = less fidgeting, more composed appearance")
        if stability < 0.80:
            lines.append(f"    ! TIP: Try to minimize head movements and maintain eye contact")
        lines.append("")
        
        # Detailed Timeline (if we have enough data)
        if len(confidence_scores) >= 10:
            lines.append("-" * 70)
            lines.append("TIMELINE BREAKDOWN")
            lines.append("-" * 70)
            lines.append("")
            
            # Split into 6 segments for detailed view
            n_segments = 6
            segment_size = len(confidence_scores) // n_segments
            segment_duration = duration / n_segments
            
            lines.append("  Time        Confidence   Trend    Notes")
            lines.append("  " + "-" * 55)
            
            prev_avg = None
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(confidence_scores)
                segment_scores = confidence_scores[start_idx:end_idx]
                segment_avg = np.mean(segment_scores)
                
                time_start = i * segment_duration
                time_end = (i + 1) * segment_duration
                
                trend = self._get_trend_arrow(prev_avg, segment_avg) if prev_avg else "   "
                note = self._get_segment_note(segment_avg, prev_avg, i, n_segments)
                
                lines.append(f"  {time_start:5.1f}s-{time_end:5.1f}s   {segment_avg:5.1%}        {trend}      {note}")
                prev_avg = segment_avg
            
            lines.append("")
        
        # Improvement Areas
        lines.append("-" * 70)
        lines.append("AREAS FOR IMPROVEMENT")
        lines.append("-" * 70)
        lines.append("")
        
        improvements = []
        if start_value < 0.60:
            improvements.append(("Opening Confidence", "Practice your introduction until it feels natural", "HIGH"))
        if engagement < 0.75:
            improvements.append(("Expressiveness", "Show more natural reactions - nod, smile, show interest", "MEDIUM"))
        if stability < 0.80:
            improvements.append(("Body Language", "Reduce fidgeting - plant your feet, relax shoulders", "MEDIUM"))
        if stress_spikes > 2:
            improvements.append(("Stress Management", "Practice deep breathing before tough questions", "HIGH"))
        if end_value < start_value:
            improvements.append(("Stamina", "Work on maintaining energy throughout longer sessions", "MEDIUM"))
        if face_detection < 0.90:
            improvements.append(("Camera Setup", "Improve lighting and camera angle for better visibility", "LOW"))
        
        if improvements:
            for area, tip, priority in improvements:
                lines.append(f"  [{priority}] {area}")
                lines.append(f"         {tip}")
                lines.append("")
        else:
            lines.append("  No major issues detected. Keep up the great work!")
            lines.append("")
        
        # Strengths
        lines.append("-" * 70)
        lines.append("YOUR STRENGTHS")
        lines.append("-" * 70)
        lines.append("")
        
        strengths = []
        if growth >= 0.15:
            strengths.append("Strong ability to warm up and build confidence during sessions")
        if end_value >= 0.75:
            strengths.append("Excellent finishing presence - you end on a high note")
        if stability >= 0.85:
            strengths.append("Composed body language with minimal fidgeting")
        if engagement >= 0.80:
            strengths.append("Engaging and expressive communication style")
        if face_detection >= 0.95:
            strengths.append("Great camera presence and positioning")
        if stress_spikes == 0:
            strengths.append("Consistent confidence without visible stress moments")
        if std_conf < 0.10:
            strengths.append("Very stable and consistent confidence throughout")
        
        if strengths:
            for strength in strengths:
                lines.append(f"  + {strength}")
            lines.append("")
        else:
            lines.append("  Keep practicing to develop your interview strengths!")
            lines.append("")
        
        # Comparison to Benchmarks
        lines.append("-" * 70)
        lines.append("BENCHMARK COMPARISON")
        lines.append("-" * 70)
        lines.append("")
        lines.append("  Your Score vs. Typical Candidates:")
        lines.append("")
        lines.append(f"  Metric           You      Typical    Status")
        lines.append("  " + "-" * 45)
        lines.append(f"  Confidence       {avg_conf:5.0%}      60%       {self._vs_benchmark(avg_conf, 0.60)}")
        lines.append(f"  Improvement      {growth:+5.0%}     +12%       {self._vs_benchmark(growth, 0.12)}")
        lines.append(f"  Expressiveness   {engagement:5.0%}      70%       {self._vs_benchmark(engagement, 0.70)}")
        lines.append(f"  Stability        {stability:5.0%}      75%       {self._vs_benchmark(stability, 0.75)}")
        lines.append("")
        
        # Final Score Card
        lines.append("-" * 70)
        lines.append("FINAL SCORE CARD")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  +{'=' * 40}+")
        lines.append(f"  |{'OVERALL INTERVIEW READINESS':^40}|")
        lines.append(f"  +{'=' * 40}+")
        lines.append(f"  |{' ' * 40}|")
        lines.append(f"  |{f'{avg_conf:.0%}':^40}|")
        lines.append(f"  |{self._get_score_label(avg_conf):^40}|")
        lines.append(f"  |{' ' * 40}|")
        lines.append(f"  +{'=' * 40}+")
        lines.append("")
        
        # Footer
        lines.append("=" * 70)
        lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("Interview Emotion Detection System v1.0")
        lines.append("=" * 70)
        
        # Save
        os.makedirs(save_path, exist_ok=True)
        session_id = data.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"{save_path}interview_report_{session_id}.txt"
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Text report saved: {filename}")
        return filename
    
    def _interpret_std(self, std: float) -> str:
        """Interpret standard deviation."""
        if std < 0.08:
            return "Very consistent"
        elif std < 0.12:
            return "Fairly stable"
        elif std < 0.18:
            return "Some variation"
        else:
            return "High variability"
    
    def _get_trend_arrow(self, prev: Optional[float], curr: float) -> str:
        """Get trend arrow between two values."""
        if prev is None:
            return "   "
        diff = curr - prev
        if diff > 0.05:
            return "[+]"
        elif diff < -0.05:
            return "[-]"
        else:
            return "[=]"
    
    def _get_segment_note(self, avg: float, prev: Optional[float], idx: int, total: int) -> str:
        """Get note for a timeline segment."""
        if idx == 0:
            if avg < 0.50:
                return "Nervous start"
            elif avg < 0.65:
                return "Cautious opening"
            else:
                return "Strong start"
        elif idx == total - 1:
            if prev and avg > prev + 0.05:
                return "Strong finish!"
            elif prev and avg < prev - 0.05:
                return "Energy dropped"
            else:
                return "Steady close"
        else:
            if prev and avg > prev + 0.08:
                return "Building momentum"
            elif prev and avg < prev - 0.08:
                return "Confidence dip"
            else:
                return "Holding steady"
    
    def _vs_benchmark(self, value: float, benchmark: float) -> str:
        """Compare value to benchmark."""
        if value >= benchmark * 1.15:
            return "ABOVE"
        elif value >= benchmark * 0.90:
            return "ON PAR"
        else:
            return "BELOW"


# Test
if __name__ == "__main__":
    import random
    
    print("Testing Insight-First Report Generator")
    print("=" * 50)
    
    # Create test data with realistic patterns
    timestamps = [i * 0.33 for i in range(180)]
    
    # Simulate nervous start, warming up, strong finish
    confidence_scores = []
    for i in range(180):
        if i < 40:
            base = 0.45 + (i / 40) * 0.15  # Start nervous, improve
        elif i < 140:
            base = 0.65 + random.uniform(-0.05, 0.05)  # Stable middle
        else:
            base = 0.70 + ((i - 140) / 40) * 0.1  # Strong finish
        confidence_scores.append(min(0.95, max(0.2, base + random.uniform(-0.05, 0.05))))
    
    sample_data = {
        'session_id': 'TEST_003',
        'session_duration': 60.0,
        'frames_analyzed': 180,
        'average_confidence': np.mean(confidence_scores),
        'engagement_score': 0.72,      # Yellow (60-79%) - weakest
        'face_stability': 0.88,        # Green (80+%)
        'face_detection_rate': 0.95,   # Green (80+%)
        'stress_spikes_count': 1,
        
        'segment_analysis': {
            'beginning': {'avg_confidence': np.mean(confidence_scores[:36])},
            'middle': {'avg_confidence': np.mean(confidence_scores[36:144])},
            'end': {'avg_confidence': np.mean(confidence_scores[144:])}
        },
        
        'improvement_areas': [
            {'recommendation': 'Practice your opening statement until it flows naturally.'}
        ],
        
        'time_series': {
            'timestamps': timestamps,
            'confidence_scores': confidence_scores
        }
    }
    
    generator = InterviewReportGenerator()
    generator.generate_visual_report(sample_data)
    generator.generate_text_report(sample_data)
    
    print("\nDone! Check outputs/ folder")
    print("\nImprovements made:")
    print("  1. Gradient fill: orange (start) -> green (end)")
    print("  2. Bolder typography throughout")
    print("  3. 'Work on this' highlight box for weak metric")
    print("  4. 'AVERAGE CONFIDENCE' label clarified")
    print("  5. Context line: '2x typical improvement!'")
    print("  6. Middle marker added to chart")
    print("  7. Bigger chart title")
