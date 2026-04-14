import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { motion, AnimatePresence } from "framer-motion"

import { ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { ThemeProvider } from "@/Components/theme-provider"
import { TooltipProvider } from "@/Components/ui/tooltip"
import { Toaster } from "@/Components/ui/sonner"

import Menu from "./Components/Menu"
import ImageLabelerCarousel from "./Components/ImageLablerCarousel"
import { UndoRedoClearContextProvider } from "./Components/UndoRedoClearContext"
import { LandingPage } from "./Components/LandingPage"
import { MyModelsPage } from "./Components/MyModelsPage"
import { InferencePage } from "./Components/InferencePage"
import { AppView, AnnotatedImage, RepresentativeImageDimensions } from "./types/Image"
import { DetectionMode } from "./Components/DetectionModeSelector"
import { AppDispatch, RootState } from "./state/store"
import { setSessionImages } from "./state/filesState/fileSlice"
import { setHardwareCapabilities } from "./state/hardwareSlice"
import {
  areObbDetectionSettingsEqual,
  DEFAULT_OBB_DETECTION_SETTINGS,
  getRecommendedObbDetectionSettings,
  normalizeObbDetectionSettings,
  summarizeRepresentativeImageDimensions,
} from "./lib/obbDetectorSettings"
import type { ObbDetectionSettings } from "./types/Image"
import {
  TutorialProvider,
  TutorialOverlay,
  WelcomeModal,
  TutorialLauncher,
  TOURS,
  useTutorial,
} from "./Components/Tutorial"

/** Restores session images+annotations from disk whenever the active species changes. */
function SessionRestorer() {
  const dispatch = useDispatch<AppDispatch>()
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId)

  useEffect(() => {
    if (!activeSpeciesId) return
    let cancelled = false

    window.api.sessionLoad(activeSpeciesId).then((result) => {
      if (cancelled || !result.ok || !result.images?.length) return

      const restored: AnnotatedImage[] = result.images.map((img, i) => {
        const safePath = img.diskPath.replace(/\\/g, "/")
        const url = `localfile:///${safePath.replace(/^\//, "")}`

        return {
          id: Date.now() + i,
          path: img.diskPath,
          url,
          filename: img.filename,
          boxes: img.boxes || [],
          selectedBoxId: null,
          history: [],
          future: [],
          speciesId: activeSpeciesId,
          diskPath: img.diskPath,
          isFinalized: img.finalized ?? false,
          hasBoxes: img.hasBoxes ?? false,
        }
      })

      dispatch(setSessionImages(restored))
    }).catch((err) => {
      console.error("Session restore failed:", err)
    })

    return () => { cancelled = true }
  }, [activeSpeciesId, dispatch])

  return null
}

/** Registers all guided tours once the TutorialProvider is mounted. */
function TourRegistrar() {
  const { registerTours } = useTutorial()
  useEffect(() => {
    registerTours(TOURS)
  }, [registerTours])
  return null
}

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

const App: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>()
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId)

  // Probe hardware capabilities once at startup and populate Redux
  useEffect(() => {
    window.api.probeHardware()
      .then((caps) => {
        const sam2Enabled = caps.device !== "cpu" && (caps.ramGb ?? 0) >= 8
        dispatch(setHardwareCapabilities({
          probed: true,
          device: caps.device,
          ramGb: caps.ramGb,
          gpuName: caps.gpuName,
          sam2Enabled,
          yoloWorldEnabled: true,
          cnnTier: caps.device === "cpu" ? "slow" : "fast",
        }))
      })
      .catch(() => {
        // Probe failed — stay at conservative defaults (all false/null)
      })
  }, [dispatch])

  // Navigation state
  const [currentView, setCurrentView] = useState<AppView>("landing")
  const [openTrainDialogOnMount, setOpenTrainDialogOnMount] = useState(false)
  const [openSchemaDialogOnMount, setOpenSchemaDialogOnMount] = useState(false)
  const [selectedModelForInference, setSelectedModelForInference] = useState<string>("")
  const [selectedInferenceSpeciesId, setSelectedInferenceSpeciesId] = useState<string>("")
  const [hasActivatedSchemaThisRun, setHasActivatedSchemaThisRun] = useState(false)

  // Workspace state
  const [color, setColor] = useState<string>(() =>
    localStorage.getItem("biovision-default-color") || "red"
  )
  const [isSwitchOn, setIsSwitchOn] = useState(false)
  const [opacity, setOpacity] = useState<number>(() => {
    const saved = localStorage.getItem("biovision-default-opacity")
    return saved ? parseInt(saved, 10) : 100
  })
  // Detection state
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("manual")
  const [obbDetectionSettings, setObbDetectionSettings] = useState<ObbDetectionSettings>(DEFAULT_OBB_DETECTION_SETTINGS)
  const [obbDetectionSettingsCustomized, setObbDetectionSettingsCustomized] = useState(false)
  const [sessionImageCountHint, setSessionImageCountHint] = useState(0)
  const [sessionRepresentativeImageDimensions, setSessionRepresentativeImageDimensions] =
    useState<RepresentativeImageDimensions | undefined>(undefined)
  const [workspaceRepresentativeImageDimensions, setWorkspaceRepresentativeImageDimensions] =
    useState<RepresentativeImageDimensions | undefined>(undefined)
  const [objectClassName, setObjectClassName] = useState("")
  const [samEnabled, setSamEnabled] = useState(false)
  const workspaceImages = useSelector((state: RootState) => state.files.fileArray)

  useEffect(() => {
    if (workspaceImages.length === 0) {
      setWorkspaceRepresentativeImageDimensions(undefined)
      return
    }
    let cancelled = false
    const sample = workspaceImages
      .slice(0, 12)
      .map((img) => img.url)
      .filter((url): url is string => typeof url === "string" && url.length > 0)

    Promise.all(
      sample.map(
        (url) =>
          new Promise<{ width: number; height: number } | null>((resolve) => {
            const image = new Image()
            image.onload = () => resolve({ width: image.naturalWidth || image.width, height: image.naturalHeight || image.height })
            image.onerror = () => resolve(null)
            image.src = url
          })
      )
    ).then((results) => {
      if (cancelled) return
      setWorkspaceRepresentativeImageDimensions(
        summarizeRepresentativeImageDimensions(
          results.filter((entry): entry is { width: number; height: number } => Boolean(entry))
        )
      )
    })

    return () => { cancelled = true }
  }, [workspaceImages])

  const effectiveImageCount = sessionImageCountHint > 0 ? sessionImageCountHint : workspaceImages.length
  const effectiveRepresentativeImageDimensions =
    workspaceRepresentativeImageDimensions ?? sessionRepresentativeImageDimensions
  const obbDetectionRecommendation = useMemo(
    () => getRecommendedObbDetectionSettings(
      effectiveImageCount,
      effectiveRepresentativeImageDimensions
    ),
    [effectiveImageCount, effectiveRepresentativeImageDimensions]
  )

  useEffect(() => {
    if (!activeSpeciesId) {
      setObbDetectionSettings(DEFAULT_OBB_DETECTION_SETTINGS)
      setObbDetectionSettingsCustomized(false)
      setSessionImageCountHint(0)
      setSessionRepresentativeImageDimensions(undefined)
      return
    }
    let cancelled = false
    window.api.sessionLoad(activeSpeciesId)
      .then((result) => {
        if (cancelled || !result?.ok) return
        const customized = Boolean(result.meta?.obbDetectionSettingsCustomized)
        setObbDetectionSettingsCustomized(customized)
        setSessionImageCountHint(typeof result.meta?.imageCount === "number" ? result.meta.imageCount : 0)
        setSessionRepresentativeImageDimensions(
          result.meta?.representativeImageDimensions as RepresentativeImageDimensions | undefined
        )
        setObbDetectionSettings(
          customized
            ? normalizeObbDetectionSettings(
              result.meta?.obbDetectionSettings as ObbDetectionSettings | undefined,
              obbDetectionRecommendation.settings
            )
            : obbDetectionRecommendation.settings
        )
      })
      .catch(() => {
        if (!cancelled) {
          setObbDetectionSettings(DEFAULT_OBB_DETECTION_SETTINGS)
          setObbDetectionSettingsCustomized(false)
          setSessionImageCountHint(0)
          setSessionRepresentativeImageDimensions(undefined)
        }
      })
    return () => { cancelled = true }
  }, [activeSpeciesId])

  useEffect(() => {
    if (!activeSpeciesId || obbDetectionSettingsCustomized) return
    if (areObbDetectionSettingsEqual(obbDetectionSettings, obbDetectionRecommendation.settings)) return
    setObbDetectionSettings(obbDetectionRecommendation.settings)
    window.api.sessionUpdateObbDetectorSettings(activeSpeciesId, {
      obbDetectionSettings: obbDetectionRecommendation.settings,
      obbDetectionSettingsCustomized: false,
    }).catch(() => {/* ignore */})
  }, [
    activeSpeciesId,
    obbDetectionRecommendation.settings,
    obbDetectionSettings,
    obbDetectionSettingsCustomized,
  ])

  const handleColorChange = (selectedColor: string) => setColor(selectedColor)
  const handleSwitchChange = () => setIsSwitchOn((prev) => !prev)
  const handleOpacityChange = (selectedOpacity: number) => setOpacity(selectedOpacity)

  // Adaptive sidebar bounds — scale with viewport width
  // 13" (~1280px): min 260, max 340 | 15" (~1600px): min 290, max 400
  // 24" (~1920px): min 320, max 460 | 27"+ (>1920px): min 360, max 520
  const getMenuBounds = (vw: number) => {
    if (vw < 1280) return { min: 260, max: 340, def: 270 }
    if (vw < 1600) return { min: 290, max: 400, def: 300 }
    if (vw < 1920) return { min: 320, max: 460, def: 340 }
    return           { min: 360, max: 520, def: 380 }
  }

  const [vw, setVw] = useState(window.innerWidth)
  const { min: MIN_MENU, max: MAX_MENU } = getMenuBounds(vw)

  useEffect(() => {
    const handleResize = () => setVw(window.innerWidth)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const pageRef = useRef<HTMLDivElement | null>(null)
  const menuWrapRef = useRef<HTMLDivElement | null>(null)

  const draggingRef = useRef(false)
  const userResizedRef = useRef(false)
  const collapsingRef = useRef(false)

  const [menuCollapsed, setMenuCollapsed] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [isDragInCollapseZone, setIsDragInCollapseZone] = useState(false)

  const [menuWidth, setMenuWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("menuWidth"))
    const initBounds = getMenuBounds(window.innerWidth)
    return Number.isFinite(saved) && saved > 0
      ? clamp(saved, initBounds.min, initBounds.max)
      : initBounds.def
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

  const COLLAPSE_THRESHOLD = MIN_MENU - 60

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      if (!pageRef.current) return

      const rect = pageRef.current.getBoundingClientRect()
      const next = e.clientX - rect.left

      if (next < COLLAPSE_THRESHOLD) {
        collapsingRef.current = true
        setIsDragInCollapseZone(true)
        setMenuWidth(clamp(next, 0, MAX_MENU))
      } else {
        collapsingRef.current = false
        setIsDragInCollapseZone(false)
        setMenuWidth(clamp(next, MIN_MENU, MAX_MENU))
      }
    }

    const onMouseUp = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      setIsDragging(false)

      if (collapsingRef.current) {
        setMenuCollapsed(true)
        setMenuWidth(0)
      }

      collapsingRef.current = false
      setIsDragInCollapseZone(false)
    }

    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [MIN_MENU, MAX_MENU, COLLAPSE_THRESHOLD])

  useLayoutEffect(() => {
    if (!menuWrapRef.current) return

    const el = menuWrapRef.current
    const overflowX = el.scrollWidth - el.clientWidth

    if (overflowX > 0) {
      const requiredWidth = clamp(Math.ceil(menuWidth + overflowX), MIN_MENU, MAX_MENU)
      if (requiredWidth > menuWidth) {
        setMenuWidth(requiredWidth)
      }
    }
  }, [menuWidth, MIN_MENU, MAX_MENU])

  const startDrag = () => {
    userResizedRef.current = true
    draggingRef.current = true
    setIsDragging(true)
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }

  const handleNavigate = (view: AppView) => {
    setCurrentView(view)
    // Reset flags when navigating away
    if (view !== "workspace") {
      setOpenTrainDialogOnMount(false)
    }
    if (view !== "inference") {
      setSelectedModelForInference("")
      setSelectedInferenceSpeciesId("")
    }
    if (view !== "landing") {
      setOpenSchemaDialogOnMount(false)
    }
  }

  const handleStartAnnotating = () => {
    setOpenSchemaDialogOnMount(true)
    setCurrentView("landing")
  }

  const handleSelectModelForInference = (selection: {
    speciesId?: string;
    modelKey?: string;
    modelKind?: "landmark" | "obb_detector";
  }) => {
    setSelectedModelForInference(selection.modelKey || "")
    setSelectedInferenceSpeciesId(selection.speciesId || "")
  }

  useEffect(() => {
    if (currentView === "workspace" && activeSpeciesId) {
      setHasActivatedSchemaThisRun(true)
    }
  }, [activeSpeciesId, currentView])

  const expandMenu = () => {
    setMenuCollapsed(false)
    userResizedRef.current = false
    setMenuWidth(getMenuBounds(vw).def)
  }

  // Render workspace (annotation view)
  const renderWorkspace = () => (
    <div
      ref={pageRef}
      className="w-screen h-screen min-w-0 min-h-0 bg-background flex overflow-hidden"
    >
      {/* Left: menu container */}
      <AnimatePresence initial={false}>
        {!menuCollapsed && (
          <motion.div
            key="sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: menuWidth, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: isDragging ? 0 : 0.25, ease: [0.32, 0, 0.67, 0] }}
            style={{ minWidth: menuCollapsed ? 0 : MIN_MENU }}
            className="shrink-0 h-full overflow-hidden border-r bg-card"
          >
            <div
              ref={menuWrapRef}
              className="h-full overflow-y-auto overflow-x-hidden bg-card scrollbar-app"
            >
              <Menu
                onOpacityChange={handleOpacityChange}
                onColorChange={handleColorChange}
                onSwitchChange={handleSwitchChange}
                onNavigateToLanding={() => handleNavigate("landing")}
                openTrainDialogOnMount={openTrainDialogOnMount}
                onTrainDialogOpened={() => setOpenTrainDialogOnMount(false)}
                detectionMode={detectionMode}
                onDetectionModeChange={setDetectionMode}
                obbDetectionSettings={obbDetectionSettings}
                obbDetectionRecommendation={obbDetectionRecommendation.summary}
                representativeImageDimensions={effectiveRepresentativeImageDimensions}
                onObbDetectionSettingsChange={(settings) => {
                  const normalized = normalizeObbDetectionSettings(settings)
                  setObbDetectionSettings(normalized)
                  setObbDetectionSettingsCustomized(true)
                  if (activeSpeciesId) {
                    window.api.sessionUpdateObbDetectorSettings(activeSpeciesId, {
                      obbDetectionSettings: normalized,
                      obbDetectionSettingsCustomized: true,
                    }).catch(() => {/* ignore */})
                  }
                }}
                className={objectClassName}
                onClassNameChange={setObjectClassName}
                samEnabled={samEnabled}
                onSamEnabledChange={setSamEnabled}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Drag handle — hidden when collapsed */}
      {!menuCollapsed && (
        <div
          onMouseDown={startDrag}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize menu"
          className={cn(
            "w-2.5 shrink-0 cursor-col-resize relative bg-transparent transition-colors",
            isDragInCollapseZone
              ? "bg-destructive/20 after:bg-destructive"
              : "hover:bg-primary/5 after:bg-border",
            "after:content-[''] after:absolute after:top-0 after:bottom-0 after:left-1/2",
            "after:-translate-x-1/2 after:w-0.5 after:opacity-95"
          )}
        />
      )}

      {/* Right: fills ALL remaining space */}
      <div
        className="relative flex-1 h-full bg-muted/30 overflow-hidden flex min-w-0"
        data-tutorial="canvas-area"
      >
        <div className="w-full h-full min-w-0 min-h-0 p-2 box-border flex">
          <div className="w-full h-full min-w-0 min-h-0 flex" data-tutorial="image-nav">
            <ImageLabelerCarousel
              color={color}
              opacity={opacity}
              isSwitchOn={isSwitchOn}
              detectionMode={detectionMode}
              obbDetectionSettings={obbDetectionSettings}
              className={objectClassName}
              samEnabled={samEnabled}
            />
          </div>
        </div>

        {/* Floating expand button — visible when sidebar is collapsed */}
        <AnimatePresence>
          {menuCollapsed && (
            <motion.div
              key="expand-btn"
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.22, ease: [0.25, 1, 0.5, 1] }}
              className="absolute left-2 top-0 bottom-0 z-50 flex items-center pointer-events-none"
            >
              <button
                onClick={expandMenu}
                aria-label="Expand sidebar"
                className={cn(
                  "pointer-events-auto flex items-center justify-center",
                  "h-8 w-8 rounded-full",
                  "bg-card border border-border/60 shadow-lg",
                  "text-muted-foreground hover:text-foreground hover:bg-primary/10 hover:border-primary/40",
                  "transition-colors"
                )}
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )

  // Render content based on current view
  const renderContent = () => {
    switch (currentView) {
      case "landing":
        return (
          <LandingPage
            onNavigate={handleNavigate}
            openSchemaDialogOnMount={openSchemaDialogOnMount}
            onSchemaDialogOpened={() => setOpenSchemaDialogOnMount(false)}
          />
        )
      case "models":
        return (
          <MyModelsPage
            onNavigate={handleNavigate}
            onSelectModelForInference={handleSelectModelForInference}
            onStartAnnotating={handleStartAnnotating}
          />
        )
      case "inference":
        return (
          <InferencePage
            onNavigate={handleNavigate}
            initialModel={selectedModelForInference}
            initialSpeciesId={selectedInferenceSpeciesId}
            hasActivatedSchemaThisRun={hasActivatedSchemaThisRun}
          />
        )
      case "workspace":
      default:
        return renderWorkspace()
    }
  }

  return (
    <ThemeProvider defaultTheme="system" storageKey="biovision-ui-theme">
      <TooltipProvider>
        <TutorialProvider>
          <TourRegistrar />
          <UndoRedoClearContextProvider>
            <SessionRestorer />
            {renderContent()}
            <Toaster position="bottom-center" />
          </UndoRedoClearContextProvider>
          <WelcomeModal />
          <TutorialOverlay />
          <TutorialLauncher />
        </TutorialProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
