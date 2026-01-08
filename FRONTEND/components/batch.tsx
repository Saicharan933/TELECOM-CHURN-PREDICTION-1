import React, { useState } from "react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Alert, AlertDescription } from "./ui"
import { Loader, Play, CheckCircle, AlertCircle } from "lucide-react"

interface BatchJobResult {
  success: boolean
  message: string
  timestamp: string
  recordsProcessed: number
}

export default function BatchJobRunner() {
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<BatchJobResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRunBatch = async () => {
    setIsRunning(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch("/api/run-batch", {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error("Failed to run batch job")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An error occurred"
      setError(errorMsg)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Monthly Batch Job</CardTitle>
          <CardDescription>Run churn prediction analysis on all customers</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-sm">Job Details</h3>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Scope</p>
                <p className="font-medium">All customers</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Processing Time</p>
                <p className="font-medium">~2-3 minutes</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Data Updated</p>
                <p className="font-medium">All metrics</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Frequency</p>
                <p className="font-medium">Monthly (1st of month)</p>
              </div>
            </div>
          </div>

          <div className="bg-muted p-4 rounded-lg border border-border">
            <h4 className="font-semibold text-sm mb-2">What this does:</h4>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              <li>Analyzes all customer data in the system</li>
              <li>Recalculates churn probability scores</li>
              <li>Updates risk category assignments</li>
              <li>Refreshes all dashboard metrics</li>
            </ul>
          </div>

          <Button onClick={handleRunBatch} disabled={isRunning} size="lg" className="w-full">
            {isRunning ? (
              <>
                <Loader className="h-4 w-4 animate-spin mr-2" />
                Running Batch Job...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Monthly Batch Job
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {result && (
        <Alert className="border-[hsl(var(--chart-2))] bg-[hsl(var(--chart-2)_/_0.05)] text-[hsl(var(--chart-2))]">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            <div className="font-semibold mb-1">{result.message}</div>
            <div className="text-sm opacity-90">
              <div>Records Processed: {result.recordsProcessed.toLocaleString()}</div>
              <div>Completed at: {new Date(result.timestamp).toLocaleString()}</div>
            </div>
          </AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <div className="font-semibold mb-1">Batch Job Failed</div>
            <div className="text-sm">{error}</div>
          </AlertDescription>
        </Alert>
      )}

      {/* Recent runs placeholder */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Batch Runs</CardTitle>
          <CardDescription>History of recent job executions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { date: "Dec 1, 2024 - 10:30 AM", status: "Success", records: 50234 },
              { date: "Nov 1, 2024 - 10:15 AM", status: "Success", records: 49876 },
              { date: "Oct 1, 2024 - 10:22 AM", status: "Success", records: 49543 },
              { date: "Sep 1, 2024 - 10:18 AM", status: "Success", records: 48921 },
            ].map((run, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg border border-border">
                <div>
                  <p className="font-medium text-sm">{run.date}</p>
                  <p className="text-sm text-muted-foreground">{run.records.toLocaleString()} records processed</p>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-[hsl(var(--chart-2))]" />
                  <span className="text-sm font-medium text-[hsl(var(--chart-2))]">{run.status}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
