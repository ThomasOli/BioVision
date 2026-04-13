import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

// ─── Tour & Step Types ──────────────────────────────────────────────────────

export type TourId = "welcome" | "annotation" | "training" | "inference";

export interface TutorialStep {
  /** CSS selector or data-tutorial attribute for the target element */
  target: string;
  /** Step headline */
  title: string;
  /** Rich description (can include JSX) */
  description: React.ReactNode;
  /** Preferred tooltip placement relative to the target */
  placement: "top" | "bottom" | "left" | "right" | "center";
  /** Which app view this step belongs to (null = any) */
  view?: string | null;
  /** If true, the spotlight expands to the full viewport (for intro/outro steps) */
  isFullscreen?: boolean;
  /** Optional action label for the primary button (defaults to "Next") */
  actionLabel?: string;
}

export interface Tour {
  id: TourId;
  name: string;
  description: string;
  steps: TutorialStep[];
  /** Icon name from lucide-react */
  icon: string;
}

// ─── Context Value ──────────────────────────────────────────────────────────

interface TutorialContextValue {
  /** Whether a tour is currently active */
  isActive: boolean;
  /** The currently running tour (null if none) */
  activeTour: Tour | null;
  /** Current step index within the active tour */
  currentStep: number;
  /** Total steps in the active tour */
  totalSteps: number;
  /** Start a specific tour */
  startTour: (tourId: TourId) => void;
  /** Advance to the next step (or finish if last) */
  nextStep: () => void;
  /** Go back one step */
  prevStep: () => void;
  /** Skip / end the current tour */
  endTour: () => void;
  /** Jump to a specific step */
  goToStep: (index: number) => void;
  /** Check if a tour has been completed before */
  isTourCompleted: (tourId: TourId) => boolean;
  /** Whether the welcome modal should be shown */
  showWelcome: boolean;
  /** Dismiss the welcome modal */
  dismissWelcome: () => void;
  /** Whether the tutorial launcher panel is open */
  launcherOpen: boolean;
  /** Toggle the tutorial launcher panel */
  setLauncherOpen: (open: boolean) => void;
  /** All available tours */
  tours: Tour[];
  /** Register tours (called once from step definitions) */
  registerTours: (tours: Tour[]) => void;
}

const TutorialContext = createContext<TutorialContextValue | null>(null);

// ─── localStorage Keys ──────────────────────────────────────────────────────

const STORAGE_PREFIX = "biovision-tutorial";
const WELCOME_DISMISSED_KEY = `${STORAGE_PREFIX}-welcome-dismissed`;
const COMPLETED_TOURS_KEY = `${STORAGE_PREFIX}-completed-tours`;

function getCompletedTours(): Set<TourId> {
  try {
    const raw = localStorage.getItem(COMPLETED_TOURS_KEY);
    if (raw) return new Set(JSON.parse(raw) as TourId[]);
  } catch { /* ignore */ }
  return new Set();
}

function saveCompletedTours(tours: Set<TourId>) {
  localStorage.setItem(COMPLETED_TOURS_KEY, JSON.stringify([...tours]));
}

function isWelcomeDismissed(): boolean {
  return localStorage.getItem(WELCOME_DISMISSED_KEY) === "true";
}

// ─── Provider ───────────────────────────────────────────────────────────────

export const TutorialProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [tours, setTours] = useState<Tour[]>([]);
  const [activeTourId, setActiveTourId] = useState<TourId | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [completedTours, setCompletedTours] = useState<Set<TourId>>(getCompletedTours);
  const [showWelcome, setShowWelcome] = useState(!isWelcomeDismissed());
  const [launcherOpen, setLauncherOpen] = useState(false);

  const activeTour = useMemo(
    () => tours.find((t) => t.id === activeTourId) ?? null,
    [tours, activeTourId]
  );

  const totalSteps = activeTour?.steps.length ?? 0;

  const registerTours = useCallback((newTours: Tour[]) => {
    setTours(newTours);
  }, []);

  const startTour = useCallback((tourId: TourId) => {
    setActiveTourId(tourId);
    setCurrentStep(0);
    setLauncherOpen(false);
  }, []);

  const endTour = useCallback(() => {
    if (activeTourId) {
      const next = new Set(completedTours);
      next.add(activeTourId);
      setCompletedTours(next);
      saveCompletedTours(next);
    }
    setActiveTourId(null);
    setCurrentStep(0);
  }, [activeTourId, completedTours]);

  const nextStep = useCallback(() => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep((s) => s + 1);
    } else {
      endTour();
    }
  }, [currentStep, totalSteps, endTour]);

  const prevStep = useCallback(() => {
    setCurrentStep((s) => Math.max(0, s - 1));
  }, []);

  const goToStep = useCallback((index: number) => {
    setCurrentStep(Math.max(0, Math.min(index, totalSteps - 1)));
  }, [totalSteps]);

  const isTourCompleted = useCallback(
    (tourId: TourId) => completedTours.has(tourId),
    [completedTours]
  );

  const dismissWelcome = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem(WELCOME_DISMISSED_KEY, "true");
  }, []);

  // Keyboard navigation while tour is active
  useEffect(() => {
    if (!activeTour) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        endTour();
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        nextStep();
      } else if (e.key === "ArrowLeft") {
        prevStep();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [activeTour, endTour, nextStep, prevStep]);

  const value = useMemo<TutorialContextValue>(
    () => ({
      isActive: !!activeTour,
      activeTour,
      currentStep,
      totalSteps,
      startTour,
      nextStep,
      prevStep,
      endTour,
      goToStep,
      isTourCompleted,
      showWelcome,
      dismissWelcome,
      launcherOpen,
      setLauncherOpen,
      tours,
      registerTours,
    }),
    [
      activeTour, currentStep, totalSteps, startTour, nextStep, prevStep,
      endTour, goToStep, isTourCompleted, showWelcome, dismissWelcome,
      launcherOpen, tours, registerTours,
    ]
  );

  return (
    <TutorialContext.Provider value={value}>
      {children}
    </TutorialContext.Provider>
  );
};

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useTutorial(): TutorialContextValue {
  const ctx = useContext(TutorialContext);
  if (!ctx) throw new Error("useTutorial must be used within TutorialProvider");
  return ctx;
}
