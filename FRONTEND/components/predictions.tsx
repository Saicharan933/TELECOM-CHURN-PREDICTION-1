import React, { useEffect, useState } from "react"
import { Input, Select, Button, Table, TableHeader, TableBody, TableRow, TableHead, TableCell, Badge } from "./ui"
import { Search, ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "../lib/utils"

export interface Prediction {
  id: string
  avgDataUsage: number
  totalComplaints: number
  monthlyBill: number
  churnProbability: number
  riskLabel: "Low" | "Medium" | "High"
}

export default function PredictionsTable() {
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
        setPredictions(data.predictions || data.data || [])
        setTotal(data.total_count || data.total || 0)
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

  const totalPages = Math.max(1, Math.ceil(total / pageSize))

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
          Showing {predictions.length > 0 ? (page - 1) * pageSize + 1 : 0} to {Math.min(page * pageSize, total)} of {total} predictions
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
