/**
 * Q&ACE API Client
 * 
 * Connects Frontend to the Unified Backend API for:
 * - Facial emotion analysis
 * - Voice emotion analysis  
 * - Text/answer quality analysis (BERT)
 * - Multimodal combined analysis
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// ============================================
// Types
// ============================================

export interface FacialAnalysisResult {
  success: boolean;
  emotions: Record<string, number>;
  dominant_emotion: string;
  confidence: number;
  face_detected: boolean;
  error?: string;
}

export interface VoiceAnalysisResult {
  success: boolean;
  emotions: Record<string, number>;
  dominant_emotion: string;
  confidence: number;
  error?: string;
}

export interface TextAnalysisResult {
  success: boolean;
  quality_score: number;
  quality_label: 'Poor' | 'Average' | 'Excellent';
  probabilities: Record<string, number>;
  feedback: string;
  error?: string;
}

export interface MultimodalAnalysisResult {
  success: boolean;
  overall_confidence: number;
  overall_emotion: string;
  facial?: FacialAnalysisResult;
  voice?: VoiceAnalysisResult;
  text?: TextAnalysisResult;
  fused_emotions: Record<string, number>;
  confidence_score: number;
  clarity_score: number;
  engagement_score: number;
  recommendations: string[];
  timestamp: string;
  error?: string;
  // Emotion timeline for detailed visualization
  emotionTimeline?: {
    samples: Array<{
      timestamp: number;
      emotions: Record<string, number>;
      confidence: number;
      faceDetected: boolean;
    }>;
    totalDuration: number;
  };
}

export interface HealthStatus {
  status: string;
  device: string;
  models: {
    facial: boolean;
    voice: boolean;
    bert: boolean;
  };
}

// ============================================
// API Client Class
// ============================================

class QnAceApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<HealthStatus> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error('API health check failed');
    }
    return response.json();
  }

  /**
   * Analyze facial emotions from image
   * @param imageBase64 - Base64 encoded image (with or without data URL prefix)
   */
  async analyzeFacial(imageBase64: string): Promise<FacialAnalysisResult> {
    const formData = new FormData();
    formData.append('image', imageBase64);

    const response = await fetch(`${this.baseUrl}/analyze/facial`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Facial analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Analyze voice emotions from audio
   * @param audioBlob - Audio blob (WAV, WebM, etc.)
   */
  async analyzeVoice(audioBlob: Blob): Promise<VoiceAnalysisResult> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await fetch(`${this.baseUrl}/analyze/voice`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Voice analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Analyze answer quality using BERT
   * @param text - Answer text
   * @param question - Optional interview question
   */
  async analyzeText(text: string, question?: string): Promise<TextAnalysisResult> {
    const response = await fetch(`${this.baseUrl}/analyze/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, question }),
    });

    if (!response.ok) {
      throw new Error(`Text analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Analyze all modalities together
   * @param options - Analysis options
   */
  async analyzeMultimodal(options: {
    image?: string;
    audio?: Blob;
    text?: string;
    question?: string;
  }): Promise<MultimodalAnalysisResult> {
    const formData = new FormData();

    if (options.image) {
      formData.append('image', options.image);
    }
    if (options.audio) {
      formData.append('audio', options.audio, 'recording.wav');
    }
    if (options.text) {
      formData.append('text', options.text);
    }
    if (options.question) {
      formData.append('question', options.question);
    }

    const response = await fetch(`${this.baseUrl}/analyze/multimodal`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Multimodal analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Capture frame from video element and convert to base64
   */
  captureFrameAsBase64(videoElement: HTMLVideoElement): string {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get canvas context');
    }
    
    ctx.drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
  }
}

// Export singleton instance
export const qnaceApi = new QnAceApiClient();

// Export class for custom instances
export { QnAceApiClient };
