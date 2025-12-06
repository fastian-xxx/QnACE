import { QnAceApiClient, qnaceApi } from '@/lib/api'

// Mock fetch globally
global.fetch = jest.fn()

const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>

describe('QnAceApiClient', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('checkHealth', () => {
    it('returns health status', async () => {
      const mockHealth = {
        status: 'ok',
        device: 'cpu',
        models: {
          facial: true,
          voice: true,
          bert: true,
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth,
      } as Response)

      const result = await qnaceApi.checkHealth()

      expect(result).toEqual(mockHealth)
      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/health')
    })

    it('throws error on failed health check', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Not Found',
      } as Response)

      await expect(qnaceApi.checkHealth()).rejects.toThrow('API health check failed')
    })
  })

  describe('analyzeFacial', () => {
    it('analyzes facial emotions', async () => {
      const mockResult = {
        success: true,
        emotions: { happy: 0.8 },
        dominant_emotion: 'happy',
        confidence: 0.85,
        face_detected: true,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      } as Response)

      const result = await qnaceApi.analyzeFacial('base64image')

      expect(result).toEqual(mockResult)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/analyze/facial',
        expect.objectContaining({
          method: 'POST',
        })
      )
    })

    it('throws error on failed analysis', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Internal Server Error',
      } as Response)

      await expect(qnaceApi.analyzeFacial('base64image')).rejects.toThrow(
        'Facial analysis failed: Internal Server Error'
      )
    })
  })

  describe('analyzeVoice', () => {
    it('analyzes voice emotions', async () => {
      const mockResult = {
        success: true,
        emotions: { calm: 0.7 },
        dominant_emotion: 'calm',
        confidence: 0.75,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      } as Response)

      const blob = new Blob(['audio data'], { type: 'audio/wav' })
      const result = await qnaceApi.analyzeVoice(blob)

      expect(result).toEqual(mockResult)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/analyze/voice',
        expect.objectContaining({
          method: 'POST',
        })
      )
    })
  })

  describe('analyzeText', () => {
    it('analyzes text quality', async () => {
      const mockResult = {
        success: true,
        quality_score: 0.85,
        quality_label: 'Excellent',
        probabilities: { excellent: 0.85 },
        feedback: 'Great answer!',
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      } as Response)

      const result = await qnaceApi.analyzeText('Test answer', 'Test question')

      expect(result).toEqual(mockResult)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/analyze/text',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: 'Test answer', question: 'Test question' }),
        })
      )
    })
  })

  describe('analyzeMultimodal', () => {
    it('analyzes all modalities', async () => {
      const mockResult = {
        success: true,
        overall_confidence: 0.8,
        overall_emotion: 'confident',
        fused_emotions: { confident: 0.8 },
        confidence_score: 0.8,
        clarity_score: 0.85,
        engagement_score: 0.75,
        recommendations: ['Keep up the good work'],
        timestamp: new Date().toISOString(),
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      } as Response)

      const result = await qnaceApi.analyzeMultimodal({
        image: 'base64image',
        audio: new Blob(['audio'], { type: 'audio/wav' }),
        text: 'Test answer',
        question: 'Test question',
      })

      expect(result).toEqual(mockResult)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/analyze/multimodal',
        expect.objectContaining({
          method: 'POST',
        })
      )
    })
  })

  describe('captureFrameAsBase64', () => {
    it('captures frame from video element', () => {
      const video = document.createElement('video')
      video.width = 640
      video.height = 480

      // Mock canvas context
      const mockToDataURL = jest.fn().mockReturnValue('data:image/jpeg;base64,test')
      const mockDrawImage = jest.fn()
      const mockGetContext = jest.fn().mockReturnValue({
        drawImage: mockDrawImage,
      })

      const originalCreateElement = document.createElement.bind(document)
      const createElementSpy = jest.spyOn(document, 'createElement')
      createElementSpy.mockImplementation((tagName) => {
        if (tagName === 'canvas') {
          const canvas = originalCreateElement('canvas') as HTMLCanvasElement
          canvas.width = 640
          canvas.height = 480
          canvas.getContext = mockGetContext as any
          canvas.toDataURL = mockToDataURL as any
          return canvas
        }
        return originalCreateElement(tagName)
      })

      const result = qnaceApi.captureFrameAsBase64(video)

      expect(result).toBe('data:image/jpeg;base64,test')
      expect(mockDrawImage).toHaveBeenCalledWith(video, 0, 0)
      
      createElementSpy.mockRestore()
    })

    it('throws error when canvas context is null', () => {
      const video = document.createElement('video')
      video.width = 640
      video.height = 480

      const originalCreateElement = document.createElement.bind(document)
      const createElementSpy = jest.spyOn(document, 'createElement')
      createElementSpy.mockImplementation((tagName) => {
        if (tagName === 'canvas') {
          const canvas = originalCreateElement('canvas') as HTMLCanvasElement
          canvas.width = 640
          canvas.height = 480
          canvas.getContext = jest.fn().mockReturnValue(null)
          return canvas
        }
        return originalCreateElement(tagName)
      })

      expect(() => qnaceApi.captureFrameAsBase64(video)).toThrow('Failed to get canvas context')
      
      createElementSpy.mockRestore()
    })
  })

  describe('custom base URL', () => {
    it('uses custom base URL', async () => {
      const customApi = new QnAceApiClient('https://custom-api.com')

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok', device: 'cpu', models: { facial: true, voice: true, bert: true } }),
      } as Response)

      await customApi.checkHealth()

      expect(mockFetch).toHaveBeenCalledWith('https://custom-api.com/health')
    })
  })
})

