export interface User {
  id: string
  email: string
  name: string
  role: string
  experienceLevel: 'entry' | 'mid' | 'senior' | 'executive'
  interviewTypes: string[]
  goals: {
    targetCompanies?: string[]
    timeline?: string
    focusAreas?: string[]
  }
  createdAt: string
}

export interface PracticeSession {
  id: string
  userId: string
  questionId: string
  question: {
    id: string
    text: string
    category: string
    difficulty: 'easy' | 'medium' | 'hard'
  }
  recordingUrl?: string
  status: 'pending' | 'recording' | 'analyzing' | 'completed' | 'failed'
  createdAt: string
  completedAt?: string
}

export interface AnalysisResult {
  facial: {
    score: number
    metrics: {
      eyeContact: number
      posture: number
      expressions: number
      confidence: number
    }
    recommendations: Array<{
      id: string
      priority: 'high' | 'medium' | 'low'
      title: string
      description: string
      details?: string
    }>
  }
  vocal: {
    score: number
    metrics: {
      tone: number
      pace: number
      clarity: number
      volume: number
    }
    recommendations: Array<{
      id: string
      priority: 'high' | 'medium' | 'low'
      title: string
      description: string
      details?: string
    }>
  }
  content: {
    score: number
    metrics: {
      structure: number
      relevance: number
      keywords: number
      completeness: number
    }
    recommendations: Array<{
      id: string
      priority: 'high' | 'medium' | 'low'
      title: string
      description: string
      details?: string
    }>
  }
  transcript: string
  keywords: string[]
  overallScore: number
}

export interface Report {
  id: string
  sessionId: string
  userId: string
  analysis: AnalysisResult
  createdAt: string
  progress?: {
    facial: number[]
    vocal: number[]
    content: number[]
    dates: string[]
  }
}

export interface Question {
  id: string
  text: string
  category: string
  difficulty: 'easy' | 'medium' | 'hard'
  tags: string[]
  createdAt: string
}

// Mock data generators
export const mockUser: User = {
  id: 'user-1',
  email: 'demo@qace.com',
  name: 'Demo User',
  role: 'Software Engineer',
  experienceLevel: 'mid',
  interviewTypes: ['technical', 'behavioral'],
  goals: {
    targetCompanies: ['Google', 'Meta', 'Amazon'],
    timeline: '3 months',
    focusAreas: ['System Design', 'Algorithms'],
  },
  createdAt: new Date().toISOString(),
}

export const mockQuestions: Question[] = [
  {
    id: 'q-1',
    text: 'Tell me about yourself and your background.',
    category: 'behavioral',
    difficulty: 'easy',
    tags: ['introduction', 'background'],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'q-2',
    text: 'Design a distributed system for handling millions of requests per second.',
    category: 'technical',
    difficulty: 'hard',
    tags: ['system-design', 'scalability'],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'q-3',
    text: 'Explain how you would optimize a slow database query.',
    category: 'technical',
    difficulty: 'medium',
    tags: ['database', 'optimization'],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'q-4',
    text: 'Describe a time when you had to work under pressure.',
    category: 'behavioral',
    difficulty: 'medium',
    tags: ['situation', 'pressure'],
    createdAt: new Date().toISOString(),
  },
]

export const mockAnalysisResult: AnalysisResult = {
  facial: {
    score: 78,
    metrics: {
      eyeContact: 82,
      posture: 75,
      expressions: 80,
      confidence: 76,
    },
    recommendations: [
      {
        id: 'f-1',
        priority: 'high',
        title: 'Improve Eye Contact',
        description: 'Try to maintain eye contact 70-80% of the time during your response.',
        details: 'Practice looking directly at the camera or interviewer. Avoid looking away for extended periods.',
      },
      {
        id: 'f-2',
        priority: 'medium',
        title: 'Enhance Posture',
        description: 'Sit up straight and maintain an open, confident posture throughout the interview.',
      },
    ],
  },
  vocal: {
    score: 85,
    metrics: {
      tone: 88,
      pace: 82,
      clarity: 87,
      volume: 83,
    },
    recommendations: [
      {
        id: 'v-1',
        priority: 'low',
        title: 'Vary Your Pace',
        description: 'Consider slowing down slightly when explaining complex concepts.',
      },
    ],
  },
  content: {
    score: 72,
    metrics: {
      structure: 75,
      relevance: 70,
      keywords: 68,
      completeness: 75,
    },
    recommendations: [
      {
        id: 'c-1',
        priority: 'high',
        title: 'Improve Structure',
        description: 'Use the STAR method (Situation, Task, Action, Result) to structure your responses.',
        details: 'Start with the situation, explain the task, describe your actions, and conclude with the result.',
      },
      {
        id: 'c-2',
        priority: 'medium',
        title: 'Include More Keywords',
        description: 'Incorporate relevant technical terms and industry keywords naturally in your responses.',
      },
    ],
  },
  transcript: 'I have been working as a software engineer for the past five years, focusing primarily on backend development and system architecture. I have experience with distributed systems and microservices...',
  keywords: ['software engineer', 'backend', 'distributed systems', 'microservices'],
  overallScore: 78,
}

export function generateMockReport(sessionId: string, userId: string): Report {
  return {
    id: `report-${Date.now()}`,
    sessionId,
    userId,
    analysis: mockAnalysisResult,
    createdAt: new Date().toISOString(),
    progress: {
      facial: [65, 70, 72, 75, 78],
      vocal: [80, 82, 83, 84, 85],
      content: [68, 70, 71, 72, 72],
      dates: [
        new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
        new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        new Date().toISOString(),
      ],
    },
  }
}

export function generateMockSession(
  userId: string,
  questionId: string
): PracticeSession {
  const question = mockQuestions.find((q) => q.id === questionId) || mockQuestions[0]
  return {
    id: `session-${Date.now()}`,
    userId,
    questionId: question.id,
    question,
    status: 'pending',
    createdAt: new Date().toISOString(),
  }
}

