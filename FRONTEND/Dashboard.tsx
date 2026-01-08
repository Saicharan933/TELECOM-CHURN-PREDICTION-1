"use client"

import type React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState, useEffect } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"
import {
  BarChart3,
  LineChartIcon,
  Zap,
  Users,
  AlertTriangle,
  TrendingUp,
  Clock,
  Search,
  ChevronLeft,
  ChevronRight,
  Moon,
  Sun,
  AlertCircle,
  CheckCircle,
  Loader,
  Play,
} from "lucide-react"
import { cn } from "@/lib/utils"

// ===== UI COMPONENTS (from shadcn/ui) =====

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

function Card({ className, ...props }: CardProps) {
  return (
    <div
      className={cn("rounded-lg border border-border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  )
}

function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
}

function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn("text-2xl font-bold tracking-tight", className)} {...props} />
}

function CardDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm text-muted-foreground", className)} {...props} />
}

function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-6 pt-0", className)} {...props} />
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost"
  size?: "sm" | "md" | "lg"
}

function Button({ className, variant = "default", size = "md", ...props }: ButtonProps) {
  const baseStyles =
    "inline-flex items-center justify-center rounded-md font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
  const variants = {
    default: "bg-primary text-primary-foreground hover:bg-primary/90",
    outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
    ghost: "hover:bg-accent hover:text-accent-foreground",
  }
  const sizes = {
    sm: "h-9 px-3 text-sm",
    md: "h-10 px-4 text-base",
    lg: "h-12 px-6 text-base",
  }

  return <button className={cn(baseStyles, variants[variant], sizes[size], className)} {...props} />
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

function Input({ className, ...props }: InputProps) {
  return (
    <input
      className={cn(
        "flex h-10 w-full rounded-md border border-input bg-input px-3 py-2 text-base text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
      {...props}
    />
  )
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

function Select({ className, ...props }: SelectProps) {
  return (
    <select
      className={cn(
        "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
      {...props}
    />
  )
}

interface TableProps extends React.TableHTMLAttributes<HTMLTableElement> {}

function Table({ className, ...props }: TableProps) {
  return <table className={cn("w-full caption-bottom text-sm", className)} {...props} />
}

function TableHeader({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <thead className={cn("border-b border-border", className)} {...props} />
}

function TableBody({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={cn("", className)} {...props} />
}

function TableRow({ className, ...props }: React.HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={cn("border-b border-border hover:bg-muted/50", className)} {...props} />
}

function TableHead({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={cn(
        "text-left align-middle font-semibold text-foreground [&:has([role=checkbox])]:pr-0 p-2",
        className,
      )}
      {...props}
    />
  )
}

function TableCell({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn("align-middle p-2 [&:has([role=checkbox])]:pr-0", className)} {...props} />
}

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors",
        variant === "default" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground",
        className,
      )}
      {...props}
    />
  )
}

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "destructive"
}

function Alert({ className, variant = "default", ...props }: AlertProps) {
  return (
    <div
      className={cn(
        "relative w-full rounded-lg border p-4",
        variant === "destructive"
          ? "border-destructive/50 bg-destructive/10 text-destructive"
          : "border-border bg-card text-foreground",
        className,
      )}
      {...props}
    />
  )
}

function AlertDescription({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-sm", className)} {...props} />
}

// ===== COMPONENTS =====

function KPICard({
  title,
  value,
  description,
  icon,
}: { title: string; value: string | number; description?: string; icon?: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <h3 className="text-sm font-medium">{title}</h3>
        {icon && <div className="text-muted-foreground">{icon}</div>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && <p className="text-xs text-muted-foreground">{description}</p>}
      </CardContent>
    </Card>
  )
}

function RiskDistributionChart() {
  const data = [
    { name: "Low Risk", count: 35156, fill: "hsl(var(--chart-2))" },
    { name: "Medium Risk", count: 10257, fill: "hsl(var(--chart-3))" },
    { name: "High Risk", count: 4821, fill: "hsl(var(--chart-4))" },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Users by Risk Category</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
            <YAxis stroke="hsl(var(--muted-foreground))" />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
              }}
              labelStyle={{ color: "hsl(var(--foreground))" }}
              cursor={{ fill: "hsl(var(--primary)/0.1)" }}
            />
            <Bar dataKey="count" fill="hsl(var(--primary))" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

function ChurnProbabilityChart() {
  const data = [
    { month: "Jan", probability: 16.2 },
    { month: "Feb", probability: 17.1 },
    { month: "Mar", probability: 17.8 },
    { month: "Apr", probability: 18.3 },
    { month: "May", probability: 18.9 },
    { month: "Jun", probability: 19.1 },
    { month: "Jul", probability: 18.7 },
    { month: "Aug", probability: 18.4 },
    { month: "Sep", probability: 18.2 },
    { month: "Oct", probability: 18.5 },
    { month: "Nov", probability: 18.6 },
    { month: "Dec", probability: 18.9 },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Average Churn Probability Trend</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" />
            <YAxis stroke="hsl(var(--muted-foreground))" />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
              }}
              labelStyle={{ color: "hsl(var(--foreground))" }}
              cursor={{ strokeDasharray: "3 3" }}
              formatter={(value) => `${value}%`}
            />
            <Line
              type="monotone"
              dataKey="probability"
              stroke="hsl(var(--chart-1))"
              dot={{ fill: "hsl(var(--chart-1))", r: 4 }}
              activeDot={{ r: 6 }}
              strokeWidth={2}
              name="Churn Probability"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

interface Prediction {
  id: string
  avgDataUsage: number
  totalComplaints: number
  monthlyBill: number
  churnProbability: number
  riskLabel: "Low" | "Medium" | "High"
}

function PredictionsTable() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [searchTerm, setSearchTerm] = useState("")
  const [riskFilter, setRiskFilter] = useState("all")
  const pageSize = 10

  const fetchPredictions = async (p: number, search: string, risk: string) => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        page: String(p),
        search,
        risk,
      })
      const response = await fetch(`/api/predictions?${params}`)
      if (response.ok) {
        const data = await response.json()
        setPredictions(data.data)
        setTotal(data.total)
      }
    } catch (error) {
      console.error("Failed to fetch predictions:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPredictions(page, searchTerm, riskFilter)
  }, [page, searchTerm, riskFilter])

  const handleSearch = (value: string) => {
    setSearchTerm(value)
    setPage(1)
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low":
        return "bg-[hsl(var(--chart-2))] text-white"
      case "Medium":
        return "bg-[hsl(var(--chart-3))] text-white"
      case "High":
        return "bg-[hsl(var(--chart-4))] text-white"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const totalPages = Math.ceil(total / pageSize)

  return (
    <div className="space-y-4">
      <div className="flex gap-4 items-end flex-wrap">
        <div className="flex-1 min-w-64">
          <label className="text-sm font-medium mb-2 block">Search by User ID</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="e.g., USER_00001"
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
        <div className="min-w-48">
          <label className="text-sm font-medium mb-2 block">Risk Level</label>
          <Select
            value={riskFilter}
            onChange={(e) => {
              setRiskFilter(e.target.value)
              setPage(1)
            }}
          >
            <option value="all">All Levels</option>
            <option value="Low">Low Risk</option>
            <option value="Medium">Medium Risk</option>
            <option value="High">High Risk</option>
          </Select>
        </div>
      </div>

      <div className="border border-border rounded-lg overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="border-b border-border">
              <TableHead className="text-foreground font-semibold">User ID</TableHead>
              <TableHead className="text-right text-foreground font-semibold">Avg Data Usage (MB)</TableHead>
              <TableHead className="text-right text-foreground font-semibold">Complaints</TableHead>
              <TableHead className="text-right text-foreground font-semibold">Monthly Bill ($)</TableHead>
              <TableHead className="text-right text-foreground font-semibold">Churn Probability</TableHead>
              <TableHead className="text-center text-foreground font-semibold">Risk Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-8">
                  <div className="animate-pulse text-muted-foreground">Loading predictions...</div>
                </TableCell>
              </TableRow>
            ) : predictions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-8">
                  <div className="text-muted-foreground">No predictions found</div>
                </TableCell>
              </TableRow>
            ) : (
              predictions.map((pred) => (
                <TableRow key={pred.id} className="border-b border-border hover:bg-muted">
                  <TableCell className="font-medium">{pred.id}</TableCell>
                  <TableCell className="text-right">{pred.avgDataUsage.toLocaleString()}</TableCell>
                  <TableCell className="text-right">{pred.totalComplaints}</TableCell>
                  <TableCell className="text-right">${pred.monthlyBill.toFixed(2)}</TableCell>
                  <TableCell className="text-right font-medium">{pred.churnProbability}%</TableCell>
                  <TableCell className="text-center">
                    <Badge className={cn("font-semibold", getRiskColor(pred.riskLabel))}>{pred.riskLabel}</Badge>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between py-4">
        <div className="text-sm text-muted-foreground">
          Showing {predictions.length > 0 ? (page - 1) * pageSize + 1 : 0} to {Math.min(page * pageSize, total)} of{" "}
          {total} predictions
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1 || loading}
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          <div className="flex items-center gap-2">
            {Array.from({ length: totalPages }).map((_, i) => {
              const pageNum = i + 1
              return (
                <Button
                  key={pageNum}
                  variant={pageNum === page ? "default" : "outline"}
                  size="sm"
                  onClick={() => setPage(pageNum)}
                  disabled={loading}
                  className="w-10"
                >
                  {pageNum}
                </Button>
              )
            })}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages || loading}
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

function ThemeToggle() {
  const [isDark, setIsDark] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const isDarkMode = document.documentElement.classList.contains("dark")
    setIsDark(isDarkMode)
  }, [])

  const toggleTheme = () => {
    const htmlElement = document.documentElement
    const newIsDark = !isDark

    if (newIsDark) {
      htmlElement.classList.add("dark")
      localStorage.setItem("theme", "dark")
    } else {
      htmlElement.classList.remove("dark")
      localStorage.setItem("theme", "light")
    }

    setIsDark(newIsDark)
  }

  if (!mounted) return null

  return (
    <Button variant="ghost" size="sm" onClick={toggleTheme} className="h-9 w-9">
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}

function Sidebar() {
  const pathname = usePathname()

  const navItems = [
    { href: "/", label: "Dashboard", icon: BarChart3 },
    { href: "/predictions", label: "Predictions", icon: LineChartIcon },
    { href: "/batch", label: "Batch Jobs", icon: Zap },
  ]

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-r border-border bg-sidebar text-sidebar-foreground flex flex-col">
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground font-bold">
            C
          </div>
          <span className="text-lg font-semibold">ChurnPredix</span>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-6">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-primary text-sidebar-primary-foreground"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          )
        })}
      </nav>

      <div className="border-t border-border p-4 flex justify-center">
        <ThemeToggle />
      </div>
    </aside>
  )
}

interface BatchJobResult {
  success: boolean
  message: string
  timestamp: string
  recordsProcessed: number
}

function BatchJobRunner() {
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

      const data: BatchJobResult = await response.json()
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

interface Metrics {
  totalUsers: number
  highRiskUsers: number
  avgChurnProbability: number
  lastBatchRun: string
}

// ===== MAIN DASHBOARD COMPONENT =====

export default function DashboardApp() {
  const [currentPage, setCurrentPage] = useState<"home" | "predictions" | "batch">("home")
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch("/api/metrics")
        if (response.ok) {
          const data = await response.json()
          setMetrics(data)
        }
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
  }, [])

  const formatTimestamp = (isoString: string) => {
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))

    if (diffHours < 1) {
      const diffMins = Math.floor(diffMs / (1000 * 60))
      return `${diffMins}m ago`
    }
    return `${diffHours}h ago`
  }

  const navItems = [
    { id: "home" as const, label: "Dashboard", icon: BarChart3 },
    { id: "predictions" as const, label: "Predictions", icon: LineChartIcon },
    { id: "batch" as const, label: "Batch Jobs", icon: Zap },
  ]

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-screen w-64 border-r border-border bg-sidebar text-sidebar-foreground flex flex-col">
        <div className="p-6 border-b border-border">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground font-bold">
              C
            </div>
            <span className="text-lg font-semibold">ChurnPredix</span>
          </div>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-6">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = currentPage === item.id
            return (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-primary text-sidebar-primary-foreground"
                    : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </button>
            )
          })}
        </nav>

        <div className="border-t border-border p-4 flex justify-center">
          <ThemeToggle />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto ml-64">
        <div className="p-8">
          <div className="space-y-8">
            {/* Home Page */}
            {currentPage === "home" && (
              <>
                <div>
                  <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
                  <p className="text-muted-foreground mt-2">
                    Monitor churn predictions and customer risk metrics in real-time
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-4">
                  <KPICard
                    title="Total Users Processed"
                    value={metrics?.totalUsers.toLocaleString() ?? "Loading..."}
                    description="Customers analyzed in system"
                    icon={<Users className="h-4 w-4" />}
                  />
                  <KPICard
                    title="High-Risk Users"
                    value={metrics?.highRiskUsers.toLocaleString() ?? "Loading..."}
                    description="Customers likely to churn"
                    icon={<AlertTriangle className="h-4 w-4" />}
                  />
                  <KPICard
                    title="Avg. Churn Probability"
                    value={metrics?.avgChurnProbability ? `${metrics.avgChurnProbability.toFixed(1)}%` : "Loading..."}
                    description="Average risk across customers"
                    icon={<TrendingUp className="h-4 w-4" />}
                  />
                  <KPICard
                    title="Last Batch Run"
                    value={metrics?.lastBatchRun ? formatTimestamp(metrics.lastBatchRun) : "Loading..."}
                    description="Most recent prediction job"
                    icon={<Clock className="h-4 w-4" />}
                  />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <RiskDistributionChart />
                  <ChurnProbabilityChart />
                </div>
              </>
            )}

            {/* Predictions Page */}
            {currentPage === "predictions" && (
              <>
                <div>
                  <h1 className="text-3xl font-bold tracking-tight">Churn Predictions</h1>
                  <p className="text-muted-foreground mt-2">
                    Browse and analyze customer churn predictions with detailed risk assessments
                  </p>
                </div>

                <PredictionsTable />
              </>
            )}

            {/* Batch Jobs Page */}
            {currentPage === "batch" && (
              <>
                <div>
                  <h1 className="text-3xl font-bold tracking-tight">Batch Jobs</h1>
                  <p className="text-muted-foreground mt-2">Manage and execute churn prediction batch processing</p>
                </div>

                <BatchJobRunner />
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
