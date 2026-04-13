import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HelpCircle } from "lucide-react";

import { cn } from "@/lib/utils";

interface ContextualHelpProps {
  /** The help text to display */
  text: string;
  /** Additional class names for the container */
  className?: string;
  /** Size of the icon (default: 14) */
  size?: number;
  /** Preferred side for the popover */
  side?: "top" | "bottom" | "left" | "right";
  /** Max width of the popover (default: 240) */
  maxWidth?: number;
}

/**
 * A small (?) icon that shows a contextual help popover on hover.
 * Use this next to section titles or controls to provide inline
 * explanations without cluttering the UI.
 */
export const ContextualHelp: React.FC<ContextualHelpProps> = ({
  text,
  className,
  size = 14,
  side = "top",
  maxWidth = 240,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const popoverPosition = (): React.CSSProperties => {
    switch (side) {
      case "top":
        return { bottom: "calc(100% + 6px)", left: "50%", transform: "translateX(-50%)" };
      case "bottom":
        return { top: "calc(100% + 6px)", left: "50%", transform: "translateX(-50%)" };
      case "left":
        return { right: "calc(100% + 6px)", top: "50%", transform: "translateY(-50%)" };
      case "right":
        return { left: "calc(100% + 6px)", top: "50%", transform: "translateY(-50%)" };
    }
  };

  return (
    <span
      className={cn("relative inline-flex items-center", className)}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <HelpCircle
        className="cursor-help text-muted-foreground/50 hover:text-muted-foreground transition-colors"
        style={{ width: size, height: size }}
      />
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.12 }}
            className="absolute z-[100] rounded-lg border border-border/60 bg-popover p-2.5 shadow-lg"
            style={{
              ...popoverPosition(),
              maxWidth,
              width: "max-content",
            }}
          >
            <p className="text-xs leading-relaxed text-popover-foreground">
              {text}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
};
