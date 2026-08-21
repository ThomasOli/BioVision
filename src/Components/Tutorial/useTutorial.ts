import { useContext } from "react";

import { TutorialContext, type TutorialContextValue } from "./TutorialContext";

/**
 * Access the tutorial controller.
 *
 * Kept in its own module so `TutorialContext.tsx` exports only components,
 * which is what React Fast Refresh needs to hot-reload the provider.
 */
export function useTutorial(): TutorialContextValue {
  const ctx = useContext(TutorialContext);
  if (!ctx) throw new Error("useTutorial must be used within TutorialProvider");
  return ctx;
}
