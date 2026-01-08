import React from "react"
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar, LineChart, Line } from "recharts"
import { Card, CardHeader, CardTitle, CardContent } from "./ui"

export function RiskDistributionChart() {
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

export function ChurnProbabilityChart() {
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
