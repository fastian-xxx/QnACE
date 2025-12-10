'use client'

import * as React from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useMediaRecorder } from '@/hooks/use-media-recorder'
import { useAnalysis } from '@/hooks/use-analysis'
import { qnaceApi } from '@/lib/api'
import { formatDuration } from '@/lib/utils'
import {
  Video,
  VideoOff,
  Mic,
  Play,
  Square,
  Pause,
  Send,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Brain,
  Eye
} from 'lucide-react'
import { cn } from '@/lib/utils'

export interface InterviewSessionProps {
  question: {
    id: string
    text: string
    category: string
    difficulty: string
  }
  onComplete?: (result: any) => void
}

interface RealTimeMetrics {
  facialEmotion: string
  facialConfidence: number
  voiceEmotion: string
  voiceConfidence: number
  faceDetected: boolean
}

export function InterviewSession({ question, onComplete }: InterviewSessionProps) {
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const [duration, setDuration] = React.useState(0)
  const [answer, setAnswer] = React.useState('')
  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const [isTranscribing, setIsTranscribing] = React.useState(false)
  const [transcriptionError, setTranscriptionError] = React.useState<string | null>(null)
  const [realTimeMetrics, setRealTimeMetrics] = React.useState<RealTimeMetrics>({
    facialEmotion: 'neutral',
    facialConfidence: 0,
    voiceEmotion: 'neutral',
    voiceConfidence: 0,
    faceDetected: false,
  })
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null)
  const analysisIntervalRef = React.useRef<NodeJS.Timeout | null>(null)
  
  // Store multiple frames captured during recording (one per 3 seconds for higher accuracy)
  const capturedFramesRef = React.useRef<string[]>([])
  // Store aggregated emotion data from all analyzed frames with timestamps
  const aggregatedEmotionsRef = React.useRef<{
    samples: Array<{
      timestamp: number  // seconds into recording
      emotions: Record<string, number>
      confidence: number
      faceDetected: boolean
    }>
    totalDuration: number
  }>({ samples: [], totalDuration: 0 })

  const {
    isAnalyzing,
    multimodalResult,
    analyzeMultimodal,
    isApiAvailable,
    error: analysisError,
    reset: resetAnalysis,
  } = useAnalysis()

  const {
    stream,
    recording,
    paused,
    blob,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    reset: resetRecording,
  } = useMediaRecorder({
    audio: true,
    video: true,
  })

  // Update video element when stream changes
  React.useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream
    }
  }, [stream])

  // Timer effect
  React.useEffect(() => {
    if (recording && !paused) {
      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 1)
      }, 1000)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [recording, paused])

  // Capture frames and analyze every 3 seconds during recording (higher frequency = better accuracy)
  React.useEffect(() => {
    if (recording && !paused && videoRef.current && isApiAvailable) {
      // Capture and analyze every 3 seconds for better accuracy
      analysisIntervalRef.current = setInterval(async () => {
        try {
          if (videoRef.current) {
            const currentTime = duration
            const base64 = qnaceApi.captureFrameAsBase64(videoRef.current)
            
            // Store the frame
            capturedFramesRef.current.push(base64)
            console.log(`📸 Frame ${capturedFramesRef.current.length} captured at ${currentTime}s`)
            
            // Analyze the frame
            const result = await qnaceApi.analyzeFacial(base64)

            if (result.success) {
              // Store sample with timestamp for weighted averaging
              aggregatedEmotionsRef.current.samples.push({
                timestamp: currentTime,
                emotions: result.emotions as Record<string, number>,
                confidence: result.confidence,
                faceDetected: result.face_detected,
              })
              aggregatedEmotionsRef.current.totalDuration = currentTime
              
              console.log(`✅ Frame ${aggregatedEmotionsRef.current.samples.length} analyzed: ${result.dominant_emotion} (${(result.confidence * 100).toFixed(0)}%)`)
              
              // Update real-time display with current frame
              setRealTimeMetrics(prev => ({
                ...prev,
                facialEmotion: result.dominant_emotion,
                facialConfidence: result.confidence * 100,
                faceDetected: result.face_detected,
              }))
            }
          }
        } catch (err) {
          console.warn('Frame analysis error:', err)
        }
      }, 3000) // Analyze every 3 seconds for higher accuracy
      
      // Capture initial frame after 1 second (camera stabilization)
      setTimeout(async () => {
        if (videoRef.current && recording) {
          try {
            const base64 = qnaceApi.captureFrameAsBase64(videoRef.current)
            capturedFramesRef.current.push(base64)
            console.log('📸 Initial frame captured')
            
            const result = await qnaceApi.analyzeFacial(base64)
            if (result.success) {
              aggregatedEmotionsRef.current.samples.push({
                timestamp: 1,
                emotions: result.emotions as Record<string, number>,
                confidence: result.confidence,
                faceDetected: result.face_detected,
              })
              
              setRealTimeMetrics(prev => ({
                ...prev,
                facialEmotion: result.dominant_emotion,
                facialConfidence: result.confidence * 100,
                faceDetected: result.face_detected,
              }))
            }
          } catch (err) {
            console.warn('Initial frame analysis error:', err)
          }
        }
      }, 1000)
      
    } else {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
        analysisIntervalRef.current = null
      }
    }

    return () => {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
      }
    }
  }, [recording, paused, isApiAvailable, duration])

  const handleStart = async () => {
    setDuration(0)
    await startRecording()
  }

  const handleStop = () => {
    // Capture final frame BEFORE stopping (stream is still active)
    if (videoRef.current && stream) {
      try {
        const finalFrame = qnaceApi.captureFrameAsBase64(videoRef.current)
        capturedFramesRef.current.push(finalFrame)
        console.log(`📸 Final frame captured. Total frames: ${capturedFramesRef.current.length}`)
      } catch (err) {
        console.warn('Failed to capture final frame:', err)
      }
    }
    stopRecording()
  }

  // Auto-transcribe when recording blob is available
  React.useEffect(() => {
    const transcribeAudio = async () => {
      if (blob && !isTranscribing && !answer && isApiAvailable) {
        setIsTranscribing(true)
        setTranscriptionError(null)
        console.log('🎙️ Starting speech transcription...')
        
        try {
          const result = await qnaceApi.transcribeSpeech(blob)
          
          if (result.success && result.text) {
            setAnswer(result.text)
            console.log('✅ Transcription complete:', result.text)
          } else if (result.error) {
            setTranscriptionError(result.error)
            console.warn('⚠️ Transcription error:', result.error)
          }
        } catch (err) {
          console.error('❌ Transcription failed:', err)
          setTranscriptionError('Failed to transcribe speech. You can type your answer manually.')
        } finally {
          setIsTranscribing(false)
        }
      }
    }
    
    transcribeAudio()
  }, [blob, isApiAvailable])

  const handleReset = () => {
    resetRecording()
    resetAnalysis()
    setDuration(0)
    setAnswer('')
    setTranscriptionError(null)
    // Reset captured frames and aggregated emotions
    capturedFramesRef.current = []
    aggregatedEmotionsRef.current = { samples: [], totalDuration: 0 }
  }

  const handleSubmit = async () => {
    if (!blob && !answer.trim()) {
      return
    }

    setIsSubmitting(true)

    try {
      // Use the best frame from captured frames
      let imageBase64: string | undefined
      
      if (capturedFramesRef.current.length > 0) {
        imageBase64 = capturedFramesRef.current[capturedFramesRef.current.length - 1]
        console.log(`📤 Submitting with ${capturedFramesRef.current.length} frames captured.`)
        console.log(`📊 Aggregated data from ${aggregatedEmotionsRef.current.samples.length} analyzed samples`)
      }
      
      // If we still have a live stream, try to capture a fresh frame
      if (videoRef.current && stream) {
        try {
          const freshFrame = qnaceApi.captureFrameAsBase64(videoRef.current)
          const quickCheck = await qnaceApi.analyzeFacial(freshFrame)
          if (quickCheck.face_detected) {
            imageBase64 = freshFrame
            console.log('📸 Using fresh frame with face detected')
          }
        } catch {
          // Use saved frame
        }
      }

      if (!imageBase64) {
        console.warn('⚠️ No image available for submission')
      }

      // Submit for full analysis
      const result = await analyzeMultimodal({
        image: imageBase64,
        audio: blob || undefined,
        text: answer || undefined,
        question: question.text,
      })

      // Enhance result with WEIGHTED AVERAGE from all samples
      const samples = aggregatedEmotionsRef.current.samples.filter(s => s.faceDetected)
      
      if (samples.length > 1 && result.facial) {
        console.log('📊 Calculating weighted average from', samples.length, 'samples')
        
        const totalDuration = aggregatedEmotionsRef.current.totalDuration || duration
        
        // Calculate weights: first 10% and last 10% get 1.5x weight (first/last impressions matter)
        const getWeight = (timestamp: number) => {
          const position = timestamp / totalDuration
          if (position <= 0.1) return 1.5  // First 10% - first impression
          if (position >= 0.9) return 1.5  // Last 10% - lasting impression
          return 1.0  // Middle 80% - normal weight
        }
        
        // Calculate weighted average emotions
        const weightedEmotions: Record<string, number> = {}
        let totalWeight = 0
        let weightedConfidence = 0
        
        for (const sample of samples) {
          const weight = getWeight(sample.timestamp)
          totalWeight += weight
          weightedConfidence += sample.confidence * weight
          
          for (const [emotion, prob] of Object.entries(sample.emotions)) {
            if (!weightedEmotions[emotion]) weightedEmotions[emotion] = 0
            weightedEmotions[emotion] += prob * weight
          }
        }
        
        // Normalize
        for (const emotion of Object.keys(weightedEmotions)) {
          weightedEmotions[emotion] /= totalWeight
        }
        weightedConfidence /= totalWeight
        
        // Find dominant emotion
        const dominantEmotion = Object.entries(weightedEmotions)
          .sort((a, b) => b[1] - a[1])[0]
        
        // Update result with weighted averages
        result.facial.emotions = weightedEmotions
        result.facial.confidence = weightedConfidence
        result.facial.dominant_emotion = dominantEmotion[0]
        
        console.log('📊 Weighted emotions:', weightedEmotions)
        console.log('📊 Dominant emotion:', dominantEmotion[0], `(${(dominantEmotion[1] * 100).toFixed(1)}%)`)
        console.log('📊 Weighted confidence:', (weightedConfidence * 100).toFixed(1) + '%')
      }

      // Add timeline data to result for detailed reporting
      const resultWithTimeline = {
        ...result,
        emotionTimeline: {
          samples: aggregatedEmotionsRef.current.samples,
          totalDuration: aggregatedEmotionsRef.current.totalDuration || duration,
        },
      }

      onComplete?.(resultWithTimeline)
    } catch (err) {
      console.error('Analysis submission failed:', err)
    } finally {
      setIsSubmitting(false)
    }
  }

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'text-green-400',
      neutral: 'text-blue-400',
      surprise: 'text-yellow-400',
      sad: 'text-purple-400',
      fear: 'text-orange-400',
      anger: 'text-red-400',
      angry: 'text-red-400',
      disgust: 'text-pink-400',
    }
    return colors[emotion] || 'text-gray-400'
  }

  const getEmotionEmoji = (emotion: string) => {
    const emojis: Record<string, string> = {
      happy: '😊',
      neutral: '😐',
      surprise: '😲',
      sad: '😢',
      fear: '😨',
      anger: '😠',
      angry: '😠',
      disgust: '🤢',
    }
    return emojis[emotion.toLowerCase()] || '😐'
  }

  // Get emotion-specific bar color for better visual feedback
  const getEmotionBarColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'bg-green-500',
      neutral: 'bg-blue-500',
      surprise: 'bg-yellow-500',
      sad: 'bg-purple-500',
      fear: 'bg-orange-500',
      anger: 'bg-red-500',
      angry: 'bg-red-500',
      disgust: 'bg-pink-500',
    }
    return colors[emotion.toLowerCase()] || 'bg-accent'
  }

  return (
    <div className="space-y-6">
      {/* Question Card */}
      <Card className="bg-gradient-to-r from-accent/10 to-accent/5 border-accent/20">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="info">{question.category}</Badge>
            <Badge variant="default">{question.difficulty}</Badge>
          </div>
          <h2 className="text-xl font-semibold">{question.text}</h2>
        </div>
      </Card>

      {/* API Status */}
      {!isApiAvailable && (
        <div className="flex items-center gap-2 rounded-lg bg-yellow-500/20 border border-yellow-500/50 p-3">
          <AlertCircle className="h-5 w-5 text-yellow-400" />
          <p className="text-sm text-yellow-400">
            Backend API not connected. Start the server with: <code className="bg-black/30 px-2 py-0.5 rounded">python api/main.py</code>
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Recording */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold flex items-center gap-2">
                  <Video className="h-5 w-5" />
                  Video Recording
                </h3>
                {recording && (
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                    <span className="text-sm font-mono">{formatDuration(duration)}</span>
                  </div>
                )}
              </div>

              {/* Video Preview */}
              <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-gradient-to-br from-gradient-purple/20 to-gradient-blue/20">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={cn('h-full w-full object-cover scale-x-[-1]', !stream && 'hidden')}
                />

                {!stream && !blob && (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-center">
                      <VideoOff className="mx-auto h-12 w-12 text-foreground/40" />
                      <p className="mt-2 text-sm text-foreground/60">Click Start to begin recording</p>
                    </div>
                  </div>
                )}

                {blob && (
                  <video
                    src={URL.createObjectURL(blob)}
                    controls
                    className="h-full w-full object-cover"
                  />
                )}

                {/* Face Detection Indicator */}
                {recording && (
                  <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-lg bg-background/80 px-3 py-1.5 backdrop-blur-sm">
                    {realTimeMetrics.faceDetected ? (
                      <CheckCircle className="h-4 w-4 text-green-400" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-400" />
                    )}
                    <span className="text-xs">
                      {realTimeMetrics.faceDetected ? 'Face Detected' : 'No Face Detected'}
                    </span>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center justify-center gap-3">
                {!recording && !blob && (
                  <Button onClick={handleStart} variant="primary" size="lg">
                    <Video className="mr-2 h-5 w-5" />
                    Start Recording
                  </Button>
                )}

                {recording && (
                  <>
                    {paused ? (
                      <Button onClick={resumeRecording} variant="primary" size="lg">
                        <Play className="mr-2 h-5 w-5" />
                        Resume
                      </Button>
                    ) : (
                      <Button onClick={pauseRecording} variant="secondary" size="lg">
                        <Pause className="mr-2 h-5 w-5" />
                        Pause
                      </Button>
                    )}
                    <Button onClick={handleStop} variant="danger" size="lg">
                      <Square className="mr-2 h-5 w-5" />
                      Stop
                    </Button>
                  </>
                )}

                {blob && (
                  <Button onClick={handleReset} variant="ghost" size="lg">
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Record Again
                  </Button>
                )}
              </div>
            </div>
          </Card>

          {/* Answer Text Input */}
          <Card>
            <div className="space-y-3">
              <h3 className="font-semibold flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Your Answer (Text)
                {isTranscribing && (
                  <span className="text-xs text-accent animate-pulse flex items-center gap-1">
                    <RefreshCw className="h-3 w-3 animate-spin" />
                    Transcribing...
                  </span>
                )}
              </h3>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder={isTranscribing ? "Transcribing your speech..." : "Your speech will be automatically transcribed here. You can edit before submitting."}
                className="w-full h-32 bg-background/50 border border-foreground/10 rounded-lg p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-accent/50"
                disabled={isTranscribing}
              />
              {transcriptionError && (
                <p className="text-xs text-red-400 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  {transcriptionError}
                </p>
              )}
              <p className="text-xs text-foreground/60">
                {answer.length} characters • {blob && !isTranscribing && answer ? 'Review and edit your transcribed answer before submitting' : 'This will be analyzed by BERT for answer quality'}
              </p>
            </div>
          </Card>
        </div>

        {/* Real-time Metrics Sidebar */}
        <div className="space-y-4">
          {/* Real-time Emotion */}
          <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10">
            <div className="space-y-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Live Analysis
              </h3>

              {recording ? (
                <div className="space-y-4">
                  {/* Facial Emotion */}
                  <div className="p-3 rounded-lg bg-background/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-foreground/60">Facial Emotion</span>
                      <span className="text-2xl transition-transform duration-300 hover:scale-125">
                        {getEmotionEmoji(realTimeMetrics.facialEmotion)}
                      </span>
                    </div>
                    <p className={cn('text-lg font-semibold capitalize transition-colors duration-300', getEmotionColor(realTimeMetrics.facialEmotion))}>
                      {realTimeMetrics.facialEmotion}
                    </p>
                    <div className="mt-2 h-2 bg-background rounded-full overflow-hidden">
                      <div
                        className={cn('h-full transition-all duration-500', getEmotionBarColor(realTimeMetrics.facialEmotion))}
                        style={{ width: `${realTimeMetrics.facialConfidence}%` }}
                      />
                    </div>
                    <p className="text-xs text-foreground/60 mt-1">
                      {realTimeMetrics.facialConfidence.toFixed(0)}% confidence
                    </p>
                  </div>

                  {/* Recording Status */}
                  <div className="flex items-center gap-2 text-sm">
                    <Mic className="h-4 w-4 text-green-400" />
                    <span>Audio Recording...</span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-foreground/60">
                  <Eye className="mx-auto h-8 w-8 mb-2 opacity-40" />
                  <p className="text-sm">Start recording to see live analysis</p>
                </div>
              )}
            </div>
          </Card>

          {/* Submit Button */}
          <Button
            onClick={handleSubmit}
            disabled={(!blob && !answer.trim()) || isSubmitting || isAnalyzing}
            variant="primary"
            size="lg"
            className="w-full"
          >
            {isSubmitting || isAnalyzing ? (
              <>
                <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="mr-2 h-5 w-5" />
                Submit for Analysis
              </>
            )}
          </Button>

          {/* Analysis Tips */}
          <Card className="bg-background/50">
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">💡 Tips</h4>
              <ul className="text-xs text-foreground/60 space-y-1">
                <li>• Look directly at the camera</li>
                <li>• Speak clearly and at moderate pace</li>
                <li>• Use the STAR method for behavioral questions</li>
                <li>• Include specific examples</li>
              </ul>
            </div>
          </Card>
        </div>
      </div>

      {/* Analysis Results */}
      {multimodalResult && (
        <Card className="bg-gradient-to-r from-green-500/10 to-blue-500/10 border-green-500/20">
          <div className="space-y-6">
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <CheckCircle className="h-6 w-6 text-green-400" />
              Analysis Complete
            </h3>

            {/* Score Cards */}
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 rounded-lg bg-background/50">
                <p className="text-3xl font-bold text-accent">{multimodalResult.confidence_score.toFixed(0)}%</p>
                <p className="text-sm text-foreground/60">Confidence</p>
              </div>
              <div className="text-center p-4 rounded-lg bg-background/50">
                <p className="text-3xl font-bold text-blue-400">{multimodalResult.clarity_score.toFixed(0)}%</p>
                <p className="text-sm text-foreground/60">Clarity</p>
              </div>
              <div className="text-center p-4 rounded-lg bg-background/50">
                <p className="text-3xl font-bold text-purple-400">{multimodalResult.engagement_score.toFixed(0)}%</p>
                <p className="text-sm text-foreground/60">Engagement</p>
              </div>
            </div>

            {/* Text Analysis */}
            {multimodalResult.text && (
              <div className="p-4 rounded-lg bg-background/50">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Answer Quality
                  </h4>
                  <Badge
                    variant={
                      multimodalResult.text.quality_label === 'Excellent' ? 'success' :
                        multimodalResult.text.quality_label === 'Average' ? 'info' : 'error'
                    }
                  >
                    {multimodalResult.text.quality_label}
                  </Badge>
                </div>
                <p className="text-sm text-foreground/80">{multimodalResult.text.feedback}</p>
              </div>
            )}

            {/* Recommendations */}
            {multimodalResult.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold mb-3">Recommendations</h4>
                <div className="space-y-2">
                  {multimodalResult.recommendations.map((rec, i) => (
                    <div key={i} className="p-3 rounded-lg bg-background/50 text-sm">
                      {rec}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Error Display */}
      {analysisError && (
        <div className="flex items-center gap-2 rounded-lg bg-red-500/20 border border-red-500/50 p-3">
          <AlertCircle className="h-5 w-5 text-red-400" />
          <p className="text-sm text-red-400">{analysisError.message}</p>
        </div>
      )}
    </div>
  )
}
