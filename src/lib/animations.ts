import { Variants, Transition } from "framer-motion"

// Spring transition for natural feel
export const springTransition: Transition = {
  type: "spring",
  stiffness: 400,
  damping: 30,
}

// Smooth ease transition
export const smoothTransition: Transition = {
  type: "tween",
  ease: [0.25, 0.1, 0.25, 1],
  duration: 0.2,
}

// Page transitions
export const fadeIn: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
}

export const slideUp: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: smoothTransition,
  },
  exit: {
    opacity: 0,
    y: 10,
    transition: { duration: 0.15 },
  },
}

export const slideDown: Variants = {
  initial: { opacity: 0, y: -10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: smoothTransition,
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: { duration: 0.15 },
  },
}

// Modal/Dialog animations
export const modalOverlay: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
}

export const modalContent: Variants = {
  initial: {
    opacity: 0,
    scale: 0.96,
    y: 10,
  },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: springTransition,
  },
  exit: {
    opacity: 0,
    scale: 0.96,
    y: 10,
    transition: { duration: 0.15 },
  },
}

// Button micro-interactions
export const buttonHover = {
  whileHover: { scale: 1.02 },
  transition: { duration: 0.15 },
}

export const buttonTap = {
  whileTap: { scale: 0.98 },
}

// Card hover effect
export const cardHover: Variants = {
  initial: {},
  hover: {
    y: -2,
    boxShadow: "0 8px 30px rgba(0, 0, 0, 0.08)",
    transition: { duration: 0.2 },
  },
}

// Stagger animations for lists
export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
  exit: {
    transition: {
      staggerChildren: 0.03,
      staggerDirection: -1,
    },
  },
}

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: {
    opacity: 1,
    y: 0,
    transition: smoothTransition,
  },
  exit: {
    opacity: 0,
    y: 8,
    transition: { duration: 0.15 },
  },
}

// Carousel image transitions (direction-aware)
export const carouselImage: Variants = {
  enter: (direction: number) => ({
    opacity: 0,
    x: direction > 0 ? 30 : -30,
  }),
  center: {
    opacity: 1,
    x: 0,
    transition: smoothTransition,
  },
  exit: (direction: number) => ({
    opacity: 0,
    x: direction > 0 ? -30 : 30,
    transition: { duration: 0.15 },
  }),
}

// Dropzone drag feedback
export const dropzoneActive: Variants = {
  initial: {
    borderColor: "hsl(var(--border))",
    backgroundColor: "hsl(var(--muted))",
  },
  active: {
    borderColor: "hsl(var(--primary))",
    backgroundColor: "hsl(var(--primary) / 0.05)",
    transition: { duration: 0.15 },
  },
}

// Popover/tooltip entrance
export const popoverAnimation: Variants = {
  initial: {
    opacity: 0,
    scale: 0.96,
    y: -4,
  },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.15 },
  },
  exit: {
    opacity: 0,
    scale: 0.96,
    y: -4,
    transition: { duration: 0.1 },
  },
}

// Sidebar stagger animation
export const sidebarContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.15,
    },
  },
}

export const sidebarItem: Variants = {
  initial: { opacity: 0, x: -10 },
  animate: {
    opacity: 1,
    x: 0,
    transition: smoothTransition,
  },
}

// Progress bar animation
export const progressBar: Variants = {
  initial: { width: 0 },
  animate: (value: number) => ({
    width: `${value}%`,
    transition: { duration: 0.3, ease: "easeOut" },
  }),
}

// Scale in animation (for icons, chips, etc.)
export const scaleIn: Variants = {
  initial: { opacity: 0, scale: 0.8 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: springTransition,
  },
  exit: {
    opacity: 0,
    scale: 0.8,
    transition: { duration: 0.1 },
  },
}

// Layout animation for smooth reordering
export const layoutTransition: Transition = {
  type: "spring",
  stiffness: 500,
  damping: 35,
}
