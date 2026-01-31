import React, { useEffect, useLayoutEffect, useRef, useState } from "react"
import { motion } from "framer-motion"

import { cn } from "@/lib/utils"
import { ThemeProvider } from "@/Components/theme-provider"
import { TooltipProvider } from "@/Components/ui/tooltip"
import { Toaster } from "@/Components/ui/sonner"

import Menu from "./Components/Menu"
import ImageLabelerCarousel from "./Components/ImageLablerCarousel"
import { UndoRedoClearContextProvider } from "./Components/UndoRedoClearContext"

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

const App: React.FC = () => {
  const [color, setColor] = useState<string>("red")
  const [isSwitchOn, setIsSwitchOn] = useState(false)
  const [opacity, setOpacity] = useState<number>(100)

  const handleColorChange = (selectedColor: string) => setColor(selectedColor)
  const handleSwitchChange = () => setIsSwitchOn((prev) => !prev)
  const handleOpacityChange = (selectedOpacity: number) => setOpacity(selectedOpacity)

  // Responsive bounds for menu
  const [isXs, setIsXs] = useState(window.innerWidth < 600)
  const MIN_MENU = isXs ? 200 : 305
  const MAX_MENU = isXs ? 360 : 680

  useEffect(() => {
    const handleResize = () => setIsXs(window.innerWidth < 600)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const pageRef = useRef<HTMLDivElement | null>(null)
  const menuWrapRef = useRef<HTMLDivElement | null>(null)

  const draggingRef = useRef(false)
  const userResizedRef = useRef(false)

  const [menuWidth, setMenuWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("menuWidth"))
    return Number.isFinite(saved) && saved > 0 ? saved : isXs ? 300 : 380
  })

  useEffect(() => {
    if (userResizedRef.current) localStorage.setItem("menuWidth", String(menuWidth))
  }, [menuWidth])

  useLayoutEffect(() => {
    if (!menuWrapRef.current) return

    const el = menuWrapRef.current
    const natural = el.scrollWidth || el.getBoundingClientRect().width

    setMenuWidth(() => {
      const saved = Number(localStorage.getItem("menuWidth"))
      const hasSaved = Number.isFinite(saved) && saved > 0
      return hasSaved ? clamp(saved, MIN_MENU, MAX_MENU) : clamp(Math.round(natural), MIN_MENU, MAX_MENU)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!menuWrapRef.current) return

    const el = menuWrapRef.current

    const ro = new ResizeObserver(() => {
      if (userResizedRef.current) return
      const natural = el.scrollWidth || el.getBoundingClientRect().width
      setMenuWidth(clamp(Math.round(natural), MIN_MENU, MAX_MENU))
    })

    ro.observe(el)
    return () => ro.disconnect()
  }, [MIN_MENU, MAX_MENU])

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      if (!pageRef.current) return

      const rect = pageRef.current.getBoundingClientRect()
      const next = e.clientX - rect.left
      setMenuWidth(clamp(next, MIN_MENU, MAX_MENU))
    }

    const onMouseUp = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
  }, [MIN_MENU, MAX_MENU])

  const startDrag = () => {
    userResizedRef.current = true
    draggingRef.current = true
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }

  return (
    <ThemeProvider defaultTheme="system" storageKey="biovision-ui-theme">
      <TooltipProvider>
        <UndoRedoClearContextProvider>
          <div
            ref={pageRef}
            className="w-screen h-screen min-w-[880px] min-h-[500px] bg-background flex overflow-auto"
          >
            {/* Left: menu container */}
            <motion.div
              animate={{ width: menuWidth }}
              transition={{ duration: 0.15, ease: "easeOut" }}
              className="shrink-0 h-full overflow-hidden border-r bg-card"
            >
              <div
                ref={menuWrapRef}
                className="h-full overflow-y-auto overflow-x-hidden bg-card"
              >
                <Menu
                  onOpacityChange={handleOpacityChange}
                  onColorChange={handleColorChange}
                  onSwitchChange={handleSwitchChange}
                />
              </div>
            </motion.div>

            {/* Drag handle */}
            <div
              onMouseDown={startDrag}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize menu"
              className={cn(
                "w-2.5 shrink-0 cursor-col-resize relative bg-transparent",
                "hover:bg-primary/5 transition-colors",
                "after:content-[''] after:absolute after:top-0 after:bottom-0 after:left-1/2",
                "after:-translate-x-1/2 after:w-0.5 after:bg-border after:opacity-95"
              )}
            />

            {/* Right: fills ALL remaining space */}
            <div className="flex-1 h-full bg-muted/30 overflow-hidden flex min-w-0">
              <div className="w-full h-full min-w-0 min-h-0 p-2 box-border flex">
                <div className="w-full h-full min-w-0 min-h-0 flex">
                  <ImageLabelerCarousel color={color} opacity={opacity} isSwitchOn={isSwitchOn} />
                </div>
              </div>
            </div>
          </div>

          <Toaster position="bottom-center" />
        </UndoRedoClearContextProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
