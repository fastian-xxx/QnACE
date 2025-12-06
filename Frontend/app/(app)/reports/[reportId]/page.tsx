'use client'

import * as React from 'react'
import { useParams } from 'next/navigation'
import { ReportSummary } from '@/components/reports/report-summary'
import { EmotionTimeline } from '@/components/reports/emotion-timeline'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ArrowLeft, FileX } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'

export default function ReportDetailPage() {
  const params = useParams()
  const [report, setReport] = React.useState<any>(null)
  const [notFound, setNotFound] = React.useState(false)

  React.useEffect(() => {
    // Load report from localStorage
    const storedReports = JSON.parse(localStorage.getItem('qace_reports') || '[]')
    const foundReport = storedReports.find((r: any) => r.id === params.reportId)

    if (foundReport) {
      setReport(foundReport)
    } else {
      setNotFound(true)
    }
  }, [params.reportId])

  if (notFound) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <FileX className="h-12 w-12 text-foreground/40 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Report Not Found</h3>
          <p className="text-foreground/60 mb-4">
            This report doesn&apos;t exist or may have been deleted.
          </p>
          <Button asChild variant="primary">
            <Link href="/reports">Back to Reports</Link>
          </Button>
        </div>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <p className="text-foreground/60">Loading report...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="flex items-center gap-4">
        <Button asChild variant="ghost" size="sm">
          <Link href="/reports">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Reports
          </Link>
        </Button>
      </div>

      <div>
        <h1 className="text-3xl font-bold mb-2">Practice Session Report</h1>
        <p className="text-foreground/60">
          Analysis completed on {new Date(report.createdAt).toLocaleDateString()}
        </p>
      </div>

      {/* Overall Score */}
      <Card>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-2">Overall Score</h2>
            <p className="text-foreground/60">
              Based on facial, vocal, and content analysis
            </p>
          </div>
          <div className="text-center">
            <div className="text-5xl font-bold gradient-text mb-2">
              {report.analysis.overallScore}%
            </div>
            <Badge
              variant={
                report.analysis.overallScore >= 80
                  ? 'success'
                  : report.analysis.overallScore >= 70
                    ? 'info'
                    : 'warning'
              }
            >
              {report.analysis.overallScore >= 80
                ? 'Excellent'
                : report.analysis.overallScore >= 70
                  ? 'Good'
                  : 'Needs Improvement'}
            </Badge>
          </div>
        </div>
      </Card>

      {/* Three-Column Summary */}
      <ReportSummary analysis={report.analysis} progress={report.progress} />

      {/* Emotion Timeline */}
      {report.emotionTimeline && report.emotionTimeline.samples && report.emotionTimeline.samples.length > 0 && (
        <Card>
          <h2 className="text-xl font-semibold mb-4">📊 Emotion Timeline</h2>
          <p className="text-foreground/60 text-sm mb-4">
            See how your emotions changed throughout your answer
          </p>
          <EmotionTimeline 
            samples={report.emotionTimeline.samples}
            totalDuration={report.emotionTimeline.totalDuration}
          />
        </Card>
      )}

      {/* Transcript */}
      {report.analysis.transcript && (
        <Card>
          <h2 className="text-xl font-semibold mb-4">Transcript</h2>
          <p className="text-foreground/70 whitespace-pre-wrap">{report.analysis.transcript}</p>
        </Card>
      )}

      {/* Keywords */}
      {report.analysis.keywords && report.analysis.keywords.length > 0 && (
        <Card>
          <h2 className="text-xl font-semibold mb-4">Key Terms Identified</h2>
          <div className="flex flex-wrap gap-2">
            {report.analysis.keywords.map((keyword: string, index: number) => (
              <Badge key={index} variant="info">
                {keyword}
              </Badge>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

