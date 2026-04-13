import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, X } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { useTutorial } from "./TutorialContext";

// ─── Geometry helpers ───────────────────────────────────────────────────────

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const PADDING = 10;
const TOOLTIP_GAP = 14;
const TOOLTIP_MAX_W = 380;

function getTargetRect(selector: string): Rect | null {
  if (selector === "__fullscreen__") return null;
  const el = document.querySelector(selector);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {
    top: r.top - PADDING,
    left: r.left - PADDING,
    width: r.width + PADDING * 2,
    height: r.height + PADDING * 2,
  };
}

function clampToViewport(x: number, y: number, w: number, h: number) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  return {
    x: Math.max(8, Math.min(x, vw - w - 8)),
    y: Math.max(8, Math.min(y, vh - h - 8)),
  };
}

// ─── SVG Spotlight ──────────────────────────────────────────────────────────

const SpotlightSvg: React.FC<{ rect: Rect | null }> = ({ rect }) => {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const r = 8; // border-radius for the cutout

  return (
    <svg
      className="pointer-events-none fixed inset-0 z-[9998]"
      width={vw}
      height={vh}
      style={{ position: "fixed", top: 0, left: 0 }}
    >
      <defs>
        <mask id="tutorial-spotlight-mask">
          <rect x={0} y={0} width={vw} height={vh} fill="white" />
          {rect && (
            <rect
              x={rect.left}
              y={rect.top}
              width={rect.width}
              height={rect.height}
              rx={r}
              ry={r}
              fill="black"
            />
          )}
        </mask>
      </defs>
      <rect
        x={0}
        y={0}
        width={vw}
        height={vh}
        fill="rgba(0, 0, 0, 0.65)"
        mask="url(#tutorial-spotlight-mask)"
      />
      {rect && (
        <rect
          x={rect.left}
          y={rect.top}
          width={rect.width}
          height={rect.height}
          rx={r}
          ry={r}
          fill="none"
          stroke="hsl(217.2 91.2% 59.8%)"
          strokeWidth={2}
          className="animate-pulse"
        />
      )}
    </svg>
  );
};

// ─── Tooltip Position Calculator ────────────────────────────────────────────

function computeTooltipPos(
  rect: Rect | null,
  placement: string,
  tooltipW: number,
  tooltipH: number,
): { x: number; y: number } {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Fullscreen / centred
  if (!rect) {
    return {
      x: (vw - tooltipW) / 2,
      y: (vh - tooltipH) / 2,
    };
  }

  let x = 0;
  let y = 0;

  switch (placement) {
    case "bottom":
      x = rect.left + rect.width / 2 - tooltipW / 2;
      y = rect.top + rect.height + TOOLTIP_GAP;
      break;
    case "top":
      x = rect.left + rect.width / 2 - tooltipW / 2;
      y = rect.top - tooltipH - TOOLTIP_GAP;
      break;
    case "right":
      x = rect.left + rect.width + TOOLTIP_GAP;
      y = rect.top + rect.height / 2 - tooltipH / 2;
      break;
    case "left":
      x = rect.left - tooltipW - TOOLTIP_GAP;
      y = rect.top + rect.height / 2 - tooltipH / 2;
      break;
    default: // center
      x = (vw - tooltipW) / 2;
      y = (vh - tooltipH) / 2;
  }

  return clampToViewport(x, y, tooltipW, tooltipH);
}

// ─── Main Overlay ───────────────────────────────────────────────────────────

export const TutorialOverlay: React.FC = () => {
  const {
    isActive,
    activeTour,
    currentStep,
    totalSteps,
    nextStep,
    prevStep,
    endTour,
    goToStep,
  } = useTutorial();

  const tooltipRef = useRef<HTMLDivElement>(null);
  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [tooltipSize, setTooltipSize] = useState({ w: TOOLTIP_MAX_W, h: 200 });

  const step = activeTour?.steps[currentStep] ?? null;

  // Measure target element and position tooltip
  const reposition = useCallback(() => {
    if (!step) return;

    const rect = getTargetRect(step.target);
    setTargetRect(rect);

    // Scroll target into view if needed
    if (rect) {
      const el = document.querySelector(step.target);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }

    const w = tooltipRef.current?.offsetWidth ?? TOOLTIP_MAX_W;
    const h = tooltipRef.current?.offsetHeight ?? 200;
    setTooltipSize({ w, h });

    const pos = computeTooltipPos(rect, step.placement, w, h);
    setTooltipPos(pos);
  }, [step]);

  // Reposition on step change and window resize
  useLayoutEffect(() => {
    reposition();
  }, [reposition, currentStep]);

  useEffect(() => {
    if (!isActive) return;
    // Small delay to let DOM update after view changes
    const timer = setTimeout(reposition, 100);
    window.addEventListener("resize", reposition);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", reposition);
    };
  }, [isActive, reposition]);

  // Re-measure after tooltip renders to get accurate height
  useLayoutEffect(() => {
    if (!tooltipRef.current || !step) return;
    const w = tooltipRef.current.offsetWidth;
    const h = tooltipRef.current.offsetHeight;
    if (w !== tooltipSize.w || h !== tooltipSize.h) {
      setTooltipSize({ w, h });
      const rect = getTargetRect(step.target);
      const pos = computeTooltipPos(rect, step.placement, w, h);
      setTooltipPos(pos);
    }
  });

  if (!isActive || !step) return null;

  const isFirst = currentStep === 0;
  const isLast = currentStep === totalSteps - 1;
  const progressPct = ((currentStep + 1) / totalSteps) * 100;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={`tutorial-overlay-${activeTour?.id}-${currentStep}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="fixed inset-0 z-[9997]"
      >
        {/* Clickable backdrop to dismiss */}
        <div
          className="fixed inset-0 z-[9997]"
          onClick={endTour}
        />

        {/* SVG Spotlight */}
        <SpotlightSvg rect={targetRect} />

        {/* Tooltip Card */}
        <motion.div
          ref={tooltipRef}
          initial={{ opacity: 0, scale: 0.95, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 8 }}
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
          className={cn(
            "fixed z-[9999] rounded-xl border border-border/60 bg-card shadow-2xl",
            "backdrop-blur-xl",
            step.isFullscreen ? "w-[420px]" : "w-[360px]",
          )}
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            maxWidth: step.isFullscreen ? 420 : TOOLTIP_MAX_W,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Progress bar */}
          <div className="h-1 w-full overflow-hidden rounded-t-xl bg-muted">
            <motion.div
              className="h-full bg-primary"
              initial={{ width: 0 }}
              animate={{ width: `${progressPct}%` }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            />
          </div>

          {/* Close button */}
          <button
            onClick={endTour}
            className={cn(
              "absolute right-3 top-3 rounded-md p-1 text-muted-foreground",
              "transition-colors hover:bg-muted hover:text-foreground"
            )}
            aria-label="Close tutorial"
          >
            <X className="h-4 w-4" />
          </button>

          {/* Content */}
          <div className="space-y-4 p-5 pt-4">
            {/* Tour name badge */}
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-primary">
                {activeTour?.name}
              </span>
              <span className="text-[11px] text-muted-foreground">
                {currentStep + 1} of {totalSteps}
              </span>
            </div>

            {/* Title */}
            <h3 className="text-base font-bold text-foreground pr-6">
              {step.title}
            </h3>

            {/* Description */}
            <div className="text-sm leading-relaxed text-muted-foreground">
              {step.description}
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-1">
                {/* Step dots */}
                {totalSteps <= 10 &&
                  Array.from({ length: totalSteps }).map((_, i) => (
                    <button
                      key={i}
                      onClick={() => goToStep(i)}
                      className={cn(
                        "h-1.5 rounded-full transition-all duration-200",
                        i === currentStep
                          ? "w-4 bg-primary"
                          : i < currentStep
                          ? "w-1.5 bg-primary/40"
                          : "w-1.5 bg-muted-foreground/20"
                      )}
                    />
                  ))}
              </div>

              <div className="flex items-center gap-2">
                {!isFirst && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={prevStep}
                    className="h-8 gap-1 text-xs"
                  >
                    <ChevronLeft className="h-3.5 w-3.5" />
                    Back
                  </Button>
                )}
                {isFirst && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={endTour}
                    className="h-8 text-xs text-muted-foreground"
                  >
                    Skip tour
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={nextStep}
                  className="h-8 gap-1 text-xs font-semibold"
                >
                  {step.actionLabel ?? (isLast ? "Finish" : "Next")}
                  {!isLast && !step.actionLabel && (
                    <ChevronRight className="h-3.5 w-3.5" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
