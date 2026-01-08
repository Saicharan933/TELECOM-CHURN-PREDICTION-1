import React, { useEffect, useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Moon, Sun, BarChart3, LineChartIcon, Zap } from "lucide-react"
import { Button } from "./ui"
import { cn } from "../lib/utils"

export function ThemeToggle() {
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

export function Sidebar() {
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
